#include <stdio.h>

void print_binary(int n) {
    for (int i=31; i>=0; i--) {
        if (n & (1 << i))
            printf("1");
        else
            printf("0");
    }
    printf("\n");
}
int main() {
    int x = 5;
    int y = x << 2;
    printf("x = %d, y = %d\n", x, y);
    print_binary(x);
    print_binary(y);
}