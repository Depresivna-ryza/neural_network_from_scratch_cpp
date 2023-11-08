#include <catch2/catch_test_macros.hpp>
#include <sstream>

#include "misc.h"
#include "neuralnetwork.h"
#include "tensor.h"

TEST_CASE("Tensor Basics") {
    Tensor t({2, 3});
    REQUIRE(t.get_shape() == std::vector<size_t>({2, 3}));

    std::vector<double> data = {1, 2, 3, 4, 5, 6};
    t.set_data(data);
    REQUIRE(t.get_data() == data);
    REQUIRE(t.get({0, 0}) == 1);
    REQUIRE(t.get({0, 1}) == 2);
    REQUIRE(t.get({0, 2}) == 3);
    REQUIRE(t.get({1, 0}) == 4);
    REQUIRE(t.get({1, 1}) == 5);
    REQUIRE(t.get({1, 2}) == 6);
}