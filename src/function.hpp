#ifndef FUNCTION_H
#define FUNCTION_H

struct Function {
    Function();
    double operator()(double x);
};

struct Sigmoid : public Function {
    double operator()(double x);
};

struct ReLU : public Function {
    static double evaluate(double x) {
        return x > 0 ? x : 0;
    }

    static double derivative(double x) {
        return x > 0 ? 1 : 0;
    }
};

#endif  // FUNCTION_H