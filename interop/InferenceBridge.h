#ifndef INFERENCE_BRIDGE_H
#define INFERENCE_BRIDGE_H

#include <string>
#include <cstdio>

#include "Ippl.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

class __attribute__((visibility("hidden"))) InferenceBridge {
    private:
        py::module_ mod_;

        template <typename F>
        void guarded(F&& f) {
            try { f(); }
            catch (py::error_already_set& e) {
                e.restore();
                PyErr_Print();
                fflush(stderr);
                ippl::Comm->abort();
            }
        }

    public:
        InferenceBridge(const std::string& dir, const std::string& name) {
            guarded([&]{
                py::module_::import("sys").attr("path").attr("insert")(0, dir);
                mod_ = py::module_::import(name.c_str());
            });
        }
        void init(int rank, int size, int dev, const py::dict& p) {
            guarded([&]{mod_.attr("init")(rank, size, dev, p);});
        }
        void infer(const py::capsule& R, const py::capsule& E, size_t n) {
            guarded([&]{ mod_.attr("infer")(R, E, n); });
        }
        void finalize() {
            guarded([&]{ mod_.attr("finalize")(); });
            mod_ = py::object();
        }
};

#endif