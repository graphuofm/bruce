//! SQL -> logical plan. The grammar stays standard SQL: SOFTAVG is an
//! ordinary three-argument function
//! `SOFTAVG(val, SIM(key_col, :param), eps)`, so sqlparser-rs handles
//! the text and this module only interprets the tree.

use sqlparser::ast::{
    Expr, FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, SelectItem, SetExpr,
    Statement, TableFactor, Value,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::logical::{LogicalPlan, Pred, ScoreExpr, SimKind};
use crate::QueryError;

/// Parse one SELECT of the supported shape:
///
/// ```sql
/// SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.1)
/// FROM movies WHERE year >= 2000 GROUP BY genre
/// ```
///
/// The optional fourth SOFTAVG argument declares an absolute error
/// budget, the contract under which the planner may substitute an
/// approximate plan: `SOFTAVG(rating, SIM(emb, :q), 0.02, 0.01)`.
/// `INF` (an identifier) is an accepted temperature and degenerates
/// to the exact uniform average via R3.
pub fn parse_query(sql: &str) -> Result<LogicalPlan, QueryError> {
    let mut stmts =
        Parser::parse_sql(&GenericDialect {}, sql).map_err(|e| QueryError::Parse(e.to_string()))?;
    if stmts.len() != 1 {
        return Err(QueryError::Parse("expected exactly one statement".into()));
    }
    let Statement::Query(q) = stmts.remove(0) else {
        return Err(QueryError::Parse("expected a SELECT".into()));
    };
    let SetExpr::Select(sel) = *q.body else {
        return Err(QueryError::Parse("expected a plain SELECT".into()));
    };

    // FROM
    if sel.from.len() != 1 {
        return Err(QueryError::Parse("expected one table in FROM".into()));
    }
    let TableFactor::Table { name, .. } = &sel.from[0].relation else {
        return Err(QueryError::Parse("expected a base table".into()));
    };
    let table = name.to_string();
    let mut plan = LogicalPlan::Scan { table };

    // WHERE (optional, v1: one comparison)
    if let Some(w) = &sel.selection {
        plan = LogicalPlan::Filter {
            pred: parse_pred(w)?,
            input: Box::new(plan),
        };
    }

    // GROUP BY (exactly one column)
    let group_col = match &sel.group_by {
        GroupByExpr::Expressions(es, _) if es.len() == 1 => ident_of(&es[0])?,
        _ => return Err(QueryError::Parse("expected GROUP BY of one column".into())),
    };

    // projection: group col + SOFTAVG(...)
    let mut softavg = None;
    for item in &sel.projection {
        if let SelectItem::UnnamedExpr(Expr::Function(f)) = item {
            if f.name.to_string().eq_ignore_ascii_case("softavg") {
                softavg = Some(parse_softavg(f)?);
            }
        }
    }
    let (val_col, score, eps, budget) =
        softavg.ok_or_else(|| QueryError::Parse("expected a SOFTAVG(...) projection".into()))?;

    Ok(LogicalPlan::SoftAgg {
        group_col,
        val_col,
        score,
        eps,
        budget,
        input: Box::new(plan),
    })
}

fn parse_softavg(
    f: &sqlparser::ast::Function,
) -> Result<(String, ScoreExpr, f64, Option<f64>), QueryError> {
    let FunctionArguments::List(list) = &f.args else {
        return Err(QueryError::Parse(
            "SOFTAVG needs (val, SIM(..), eps[, budget])".into(),
        ));
    };
    let args: Vec<&FunctionArg> = list.args.iter().collect();
    if args.len() != 3 && args.len() != 4 {
        return Err(QueryError::Parse("SOFTAVG takes 3 or 4 arguments".into()));
    }
    let val_col = ident_of(arg_expr(args[0])?)?;
    let score = parse_sim(arg_expr(args[1])?)?;
    let eps = eps_of(arg_expr(args[2])?)?;
    let budget = if args.len() == 4 {
        Some(num_of(arg_expr(args[3])?)?)
    } else {
        None
    };
    Ok((val_col, score, eps, budget))
}

/// A temperature literal: a number, or the identifier `INF`.
///
/// Pinned semantics (tests/frontend_fuzz.rs): a numeric literal
/// beyond f64 range (e.g. `1e999`) saturates to +inf under
/// `str::parse::<f64>` and therefore degenerates to the exact
/// uniform-mean endpoint via R3 — it is NOT a parse error.
fn eps_of(e: &Expr) -> Result<f64, QueryError> {
    if let Expr::Identifier(i) = e {
        if i.value.eq_ignore_ascii_case("inf") || i.value.eq_ignore_ascii_case("infinity") {
            return Ok(f64::INFINITY);
        }
    }
    num_of(e)
}

fn parse_sim(e: &Expr) -> Result<ScoreExpr, QueryError> {
    let Expr::Function(f) = e else {
        return Err(QueryError::Parse(
            "second SOFTAVG argument must be SIM(..)".into(),
        ));
    };
    let fname = f.name.to_string().to_ascii_lowercase();
    let kind = match fname.as_str() {
        "sim" | "dot" => SimKind::Dot,
        "negsq" => SimKind::NegSq,
        "indicator" => SimKind::Indicator,
        _ => return Err(QueryError::Parse(format!("unknown score function {fname}"))),
    };
    let FunctionArguments::List(list) = &f.args else {
        return Err(QueryError::Parse("SIM needs (key_col, :param)".into()));
    };
    let args: Vec<&FunctionArg> = list.args.iter().collect();
    if args.len() != 2 {
        return Err(QueryError::Parse("SIM takes exactly 2 arguments".into()));
    }
    let key_col = ident_of(arg_expr(args[0])?)?;
    let param = match arg_expr(args[1])? {
        Expr::Value(Value::Placeholder(p)) => p.trim_start_matches(':').to_string(),
        other => return Err(QueryError::Parse(format!("expected :param, got {other}"))),
    };
    Ok(ScoreExpr {
        key_col,
        param,
        kind,
    })
}

fn parse_pred(e: &Expr) -> Result<Pred, QueryError> {
    use sqlparser::ast::BinaryOperator as B;
    let Expr::BinaryOp { left, op, right } = e else {
        return Err(QueryError::Parse(format!("unsupported WHERE: {e}")));
    };
    let col = ident_of(left)?;
    let val = num_of(right)?;
    match op {
        B::GtEq => Ok(Pred::GtEq(col, val)),
        B::Eq => Ok(Pred::Eq(col, val)),
        _ => Err(QueryError::Parse(format!("unsupported operator {op:?}"))),
    }
}

fn arg_expr(a: &FunctionArg) -> Result<&Expr, QueryError> {
    match a {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Ok(e),
        _ => Err(QueryError::Parse("unsupported function argument".into())),
    }
}

fn ident_of(e: &Expr) -> Result<String, QueryError> {
    match e {
        Expr::Identifier(i) => Ok(i.value.clone()),
        _ => Err(QueryError::Parse(format!(
            "expected a column name, got {e}"
        ))),
    }
}

fn num_of(e: &Expr) -> Result<f64, QueryError> {
    match e {
        Expr::Value(Value::Number(n, _)) => n
            .parse::<f64>()
            .map_err(|_| QueryError::Parse(format!("bad number {n}"))),
        _ => Err(QueryError::Parse(format!("expected a number, got {e}"))),
    }
}
