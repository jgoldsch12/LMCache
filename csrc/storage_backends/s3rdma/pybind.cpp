/*
 * Copyright (c) 2026 Dell Inc. or its subsidiaries.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file pybind.cpp
 * @brief pybind11 bindings for S3RdmaClient.
 *
 * Exposes the lmcache_s3rdma Python module with the S3RdmaClient class
 * and associated constants.  All methods release the GIL to allow
 * concurrent Python operations during blocking RDMA calls.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "s3rdma_client.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_s3rdma, m) {
    m.doc() = "LMCache S3 RDMA client bindings (libibverbs, NVIDIA+AMD)";

    // Constants
    m.attr("S3RDMA_SUCCESS") = lmcache::s3rdma::S3RDMA_SUCCESS;
    m.attr("S3RDMA_FAIL") = lmcache::s3rdma::S3RDMA_FAIL;
    m.attr("S3RDMA_MAX_MEMORY_REG_SIZE") = lmcache::s3rdma::S3RDMA_MAX_MEMORY_REG_SIZE;
    m.attr("S3RDMA_MEMORY_SYSTEM") = lmcache::s3rdma::S3RDMA_MEMORY_SYSTEM;
    m.attr("S3RDMA_MEMORY_GPU") = lmcache::s3rdma::S3RDMA_MEMORY_GPU;

    // S3RdmaConfig
    py::class_<lmcache::s3rdma::S3RdmaConfig>(m, "S3RdmaConfig")
        .def(py::init<>())
        .def(py::init([](const std::string& local_ip, int rdma_port,
                         const std::string& connection_mode) {
            lmcache::s3rdma::S3RdmaConfig cfg;
            cfg.local_ip = local_ip;
            cfg.rdma_port = rdma_port;
            cfg.connection_mode = connection_mode;
            return cfg;
        }),
             py::arg("local_ip") = "0.0.0.0",
             py::arg("rdma_port") = 7471,
             py::arg("connection_mode") = "RC")
        .def_readwrite("local_ip", &lmcache::s3rdma::S3RdmaConfig::local_ip)
        .def_readwrite("rdma_port", &lmcache::s3rdma::S3RdmaConfig::rdma_port)
        .def_readwrite("connection_mode",
                       &lmcache::s3rdma::S3RdmaConfig::connection_mode);

    // S3RdmaClient
    py::class_<lmcache::s3rdma::S3RdmaClient>(m, "S3RdmaClient")
        .def(py::init<const lmcache::s3rdma::S3RdmaConfig&>(),
             py::arg("config"),
             py::call_guard<py::gil_scoped_release>(),
             "Open RDMA device and optionally start RC listener.")

        .def("register_region",
             &lmcache::s3rdma::S3RdmaClient::register_region,
             py::arg("ptr"), py::arg("size"), py::arg("is_gpu") = false,
             py::call_guard<py::gil_scoped_release>(),
             "Register a contiguous memory region for RDMA.")

        .def("register_pages",
             &lmcache::s3rdma::S3RdmaClient::register_pages,
             py::arg("base"), py::arg("total_size"), py::arg("page_size"),
             py::arg("is_gpu") = false,
             py::call_guard<py::gil_scoped_release>(),
             "Register per-page MRs covering a contiguous buffer.")

        .def("deregister_region",
             &lmcache::s3rdma::S3RdmaClient::deregister_region,
             py::arg("ptr"),
             py::call_guard<py::gil_scoped_release>(),
             "Deregister a previously registered memory region.")

        .def("deregister_all",
             &lmcache::s3rdma::S3RdmaClient::deregister_all,
             py::call_guard<py::gil_scoped_release>(),
             "Deregister all registered memory regions.")

        .def("prepare_put",
             &lmcache::s3rdma::S3RdmaClient::prepare_put,
             py::arg("data_ptr"), py::arg("size"),
             py::call_guard<py::gil_scoped_release>(),
             "Generate x-rdma-info descriptor for PUT (RDMA READ by server).")

        .def("prepare_get",
             &lmcache::s3rdma::S3RdmaClient::prepare_get,
             py::arg("data_ptr"), py::arg("size"),
             py::call_guard<py::gil_scoped_release>(),
             "Generate x-rdma-info descriptor for GET (RDMA WRITE by server).")

        .def("is_connected",
             &lmcache::s3rdma::S3RdmaClient::is_connected,
             py::call_guard<py::gil_scoped_release>(),
             "Check if the RDMA device is ready.")

        .def("close",
             &lmcache::s3rdma::S3RdmaClient::close,
             py::call_guard<py::gil_scoped_release>(),
             "Release all RDMA resources.");
}
