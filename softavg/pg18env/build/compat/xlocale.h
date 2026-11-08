/* pg18env shim: the conda PG 18.6 build defined HAVE_XLOCALE_H (conda sysroot glibc 2.17);
 * glibc >= 2.26 (Ubuntu 22.04: 2.35) dropped xlocale.h and declares locale_t in locale.h. */
#include <locale.h>
