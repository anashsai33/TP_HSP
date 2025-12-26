#include "wallet.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

int main(int argc, char** argv) {
    std::string mode = "virtual";
    if (argc >= 2) mode = argv[1];

    Wallet w(10);

    if (mode == "seq") {
        std::cout << "=== Mode: seq ===\n";
        std::cout << "Initial: " << w.balance_physical() << "\n";

        w.credit(5);
        w.debit(12);

        std::cout << "Final: " << w.balance_physical() << "\n";
        return 0;
    }

    if (mode == "threads") {
        std::cout << "=== Mode: threads ===\n";
        std::cout << "Initial: " << w.balance_physical() << "\n";

        std::thread t1([&]() { w.credit(5); });
        std::thread t2([&]() { w.debit(12); });

        if (t1.joinable()) t1.join();
        if (t2.joinable()) t2.join();

        std::cout << "Final: " << w.balance_physical() << "\n";
        return 0;
    }

    if (mode == "mutex") {
        std::cout << "=== Mode: mutex ===\n";
        std::cout << "Initial: " << w.balance_physical() << "\n";

        std::thread t1([&]() { w.credit(5); });
        std::thread t2([&]() { w.debit(12); });

        if (t1.joinable()) t1.join();
        if (t2.joinable()) t2.join();

        std::cout << "Final: " << w.balance_physical() << "\n";
        return 0;
    }

    std::cout << "=== Mode: virtual ===\n";
    std::cout << "Initial balance (virtual): " << w.balance() << "\n";

    std::cout << "\n[SALE] virtual +5\n";
    w.virtual_credit(5);

    std::cout << "\n[PURCHASE] virtual -12\n";
    bool ok = w.virtual_debit(12);
    if (!ok) std::cout << "purchase refused\n";

    std::cout << "\n(main) game keeps running...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    w.wait_all();
    std::cout << "\nFinal balance (virtual): " << w.balance() << "\n";
    return 0;
}
