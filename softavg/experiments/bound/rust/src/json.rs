//! Minimal JSON writer (no external crates).
pub enum J {
    Null,
    B(bool),
    N(f64),
    I(i64),
    S(String),
    A(Vec<J>),
    O(Vec<(String, J)>),
}

impl J {
    pub fn obj(kv: Vec<(&str, J)>) -> J {
        J::O(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }
    pub fn nums(x: &[f64]) -> J {
        J::A(x.iter().map(|&v| J::N(v)).collect())
    }
    pub fn render(&self) -> String {
        let mut s = String::new();
        self.w(&mut s);
        s
    }
    fn w(&self, s: &mut String) {
        match self {
            J::Null => s.push_str("null"),
            J::B(b) => s.push_str(if *b { "true" } else { "false" }),
            J::N(x) => {
                if x.is_finite() {
                    s.push_str(&format!("{x:e}"))
                } else if x.is_nan() {
                    s.push_str("null")
                } else if *x > 0.0 {
                    s.push_str("\"inf\"")
                } else {
                    s.push_str("\"-inf\"")
                }
            }
            J::I(i) => s.push_str(&i.to_string()),
            J::S(t) => {
                s.push('"');
                for c in t.chars() {
                    match c {
                        '"' => s.push_str("\\\""),
                        '\\' => s.push_str("\\\\"),
                        '\n' => s.push_str("\\n"),
                        _ => s.push(c),
                    }
                }
                s.push('"');
            }
            J::A(v) => {
                s.push('[');
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    x.w(s);
                }
                s.push(']');
            }
            J::O(v) => {
                s.push('{');
                for (i, (k, x)) in v.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    J::S(k.clone()).w(s);
                    s.push(':');
                    x.w(s);
                }
                s.push('}');
            }
        }
    }
}
