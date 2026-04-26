#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "turboquant.hpp"

namespace py = pybind11;

PYBIND11_MODULE(turboquant_pybind, m) {
    py::class_<TurboQuant>(m, "TurboQuant")
        .def(py::init<int, int>(), py::arg("dim"), py::arg("b") = 2)
        .def("compress_mse", &TurboQuant::compress_mse,
             "Compress vector using MSE quantization (Stages 1 & 2)")
        .def("apply_qjl_residual", &TurboQuant::apply_qjl_residual,
             "Apply QJL residual correction (Stage 3)")
        .def("decompress_with_qjl", &TurboQuant::decompress_with_qjl,
             "Decompress quantized indices with QJL correction",
             py::arg("indices"), py::arg("qjl_delta") = 0.1f)
        .def_readwrite("d", &TurboQuant::d)
        .def_readwrite("centroids", &TurboQuant::centroids)
        .def_readwrite("qjl_signs", &TurboQuant::qjl_signs);

    m.doc() = "TurboQuant: Fast vector quantization for embeddings";
}
