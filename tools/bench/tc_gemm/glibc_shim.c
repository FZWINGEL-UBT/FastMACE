/* glibc 2.38 renamed strtoul for C23 callers (__isoc23_strtoul); the static libstdc++ of a newer build machine
   references it. Older distributions (Modal's Debian image, glibc 2.36) lack it, so the library defines it here,
   hidden, so that it never interposes the C library's own.
   Compiled as plain C (gnu17, no _GNU_SOURCE), where strtoul is not redirected, so this calls the old symbol. */
#include <stdlib.h>

__attribute__((visibility("hidden"))) unsigned long __isoc23_strtoul(const char *s, char **end, int base) { return strtoul(s, end, base); }
