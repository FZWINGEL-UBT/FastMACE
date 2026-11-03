/* Old-glibc shims for the probe on Ubuntu 22.04 (glibc 2.35): C23 strtoul and arc4random (glibc 2.38, 2.36). */
#include <stdint.h>
#include <stdlib.h>
#include <sys/random.h>
unsigned long __isoc23_strtoul(const char *s, char **end, int base) { return strtoul(s, end, base); }
uint32_t arc4random(void) { uint32_t v = 0; while (getrandom(&v, sizeof v, 0) != (ssize_t)sizeof v) {} return v; }
