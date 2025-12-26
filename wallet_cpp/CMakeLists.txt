cmake_minimum_required(VERSION 3.16)
project(wallet_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(wallet_demo
    src/main.cpp
    src/wallet.cpp
)

target_include_directories(wallet_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
