#include <iostream>
#include <cstdint>  // For uint8_t

int main() {
    unsigned high4bits, low4bits;

    // Get two 4-bit integers from the user
    std::cout << "Enter the high 4 bits (0-15): ";
    std::cin >> high4bits;
    std::cout << "Enter the low 4 bits (0-15): ";
    std::cin >> low4bits;

    // Check if the values are within the 4-bit range
    if (high4bits > 15 || low4bits > 15) {
        std::cerr << "Error: Values must be between 0 and 15." << std::endl;
        return 1;
    }

    // Combine the high and low 4 bits into a single 8-bit integer
    uint8_t int8 = (high4bits << 4) | low4bits;

    // Extract and print the combined 8-bit value, high and low 4 bits
    std::cout << "Combined 8-bit value: 0x" << std::hex << static_cast<int>(int8) << std::endl;
    std::cout << "Extracted high 4 bits: 0x" << std::hex << (int8 >> 4) << " (" << std::dec << (int8 >> 4) << ")" << std::endl;
    std::cout << "Extracted low 4 bits: 0x" << std::hex << (int8 & 0b00001111) << " (" << std::dec << (int8 & 0b00001111) << ")" << std::endl;

    return 0;
}
