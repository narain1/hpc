#include <stdio.h>
#include <stdlib.h>

void print_binary(int n) {
    for (int i=31; i>=0; i--) {
        if (n & (1 << i))
            printf("1");
        else
            printf("0");
    }
    printf("\n");
}

void print_binary_long(long n) {
    for (int i=63; i>=0; i--) {
        if (n & (1 << i))
            printf("1");
        else
            printf("0");
    }
    printf("\n");
}

int main() {
    int x = 5;
    long y = 0;

    y = x << 32;
    printf("%s\n", itoa(x, 2));
    print_binary(x);
    print_binary_long(y);

}