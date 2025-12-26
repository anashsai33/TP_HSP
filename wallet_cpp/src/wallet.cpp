#include "wallet.hpp"

#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

Wallet::Wallet(unsigned int start)
    : rupees(start), virtual_rupees(start) {}

void Wallet::credit(unsigned int val) {
    for (unsigned int i = 0; i < val; ++i) {
        {
            std::lock_guard<std::mutex> lock(m_rupees);
            ++rupees;
            std::cout << "+1 rupee (rupees=" << rupees << ")\n";
        }
        std::this_thread::sleep_for(100ms);
    }
}

void Wallet::debit(unsigned int val) {
    for (unsigned int i = 0; i < val; ++i) {
        {
            std::lock_guard<std::mutex> lock(m_rupees);
            if (rupees == 0) {
                std::cout << "can't debit: 0 rupee\n";
            } else {
                --rupees;
                std::cout << "-1 rupee (rupees=" << rupees << ")\n";
            }
        }
        std::this_thread::sleep_for(100ms);
    }
}

unsigned int Wallet::balance_physical() {
    std::lock_guard<std::mutex> lock(m_rupees);
    return rupees;
}

unsigned int Wallet::balance() {
    std::lock_guard<std::mutex> lock(m_virtual);
    return virtual_rupees;
}

bool Wallet::virtual_credit(unsigned int val) {
    {
        std::lock_guard<std::mutex> lock(m_virtual);
        virtual_rupees += val;
    }

    std::thread t([this, val]() { credit(val); });
    {
        std::lock_guard<std::mutex> lock(m_workers);
        workers.push_back(std::move(t));
    }
    return true;
}

bool Wallet::virtual_debit(unsigned int val) {
    {
        std::lock_guard<std::mutex> lock(m_virtual);
        if (virtual_rupees < val) return false;
        virtual_rupees -= val;
    }

    std::thread t([this, val]() { debit(val); });
    {
        std::lock_guard<std::mutex> lock(m_workers);
        workers.push_back(std::move(t));
    }
    return true;
}

void Wallet::wait_all() {
    std::lock_guard<std::mutex> lock(m_workers);
    for (auto &t : workers) {
        if (t.joinable()) t.join();
    }
    workers.clear();
}

Wallet::~Wallet() {
    wait_all();
}
