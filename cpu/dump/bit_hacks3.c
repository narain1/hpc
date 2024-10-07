#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main() {
    int a = 1;
    uint8_t b = 2;
    for (int i=0; i<10; i++) {
        a = a << b;
        printf("a = %d\n", a);
    }
}