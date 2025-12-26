#pragma once

#include <mutex>
#include <thread>
#include <vector>

class Wallet {
private:
    unsigned int rupees;
    std::mutex m_rupees;

    unsigned int virtual_rupees;
    std::mutex m_virtual;

    std::vector<std::thread> workers;
    std::mutex m_workers;

public:
    explicit Wallet(unsigned int start = 0);

    void credit(unsigned int val);
    void debit(unsigned int val);

    unsigned int balance_physical();
    unsigned int balance();

    bool virtual_credit(unsigned int val);
    bool virtual_debit(unsigned int val);

    void wait_all();
    ~Wallet();
};
