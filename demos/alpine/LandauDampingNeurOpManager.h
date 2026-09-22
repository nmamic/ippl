#ifndef IPPL_LANDAU_DAMPING_NEUROP_MANAGER_H
#define IPPL_LANDAU_DAMPING_NEUROP_MANAGER_H

#include "LandauDampingManager.h"
#include "IpplDLPack.h"

#include <pybind11/stl.h>

#include "InferenceBridge.h"

namespace py = pybind11;

template <typename T, unsigned Dim>
class LandauDampingNeurOpManager : public LandauDampingManager<T, Dim> {
    private:
        InferenceBridge* bridge;

    public:
        using Manager_t = LandauDampingManager<T, Dim>;

        LandauDampingNeurOpManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_,
                               double lbt_, std::string& solver_, std::string& stepMethod_,
                               std::vector<std::string>& preconditioner_params_,
                               InferenceBridge* bridge_)
        : Manager_t(totalP_, nt_, nr_, lbt_, solver_, stepMethod_, preconditioner_params_), bridge(bridge_) {}

        void pre_run() override {
            Manager_t::pre_run();

            py::dict params;
            params["dim"]            = Dim;
            params["totalP"]         = this->totalP_m;
            params["alpha"]          = this->alpha_m;
            params["dt"]             = this->dt_m;
            params["Q_total"]        = this->Q_m;
            params["q_per_particle"] = this->Q_m / static_cast<double>(this->totalP_m);

            std::vector<double> L(Dim), kw(Dim);
            for (unsigned d = 0; d < Dim; ++d) {
                L[d]  = this->rmax_m[d] - this->rmin_m[d];
                kw[d] = this->kw_m[d];
            }
            params["L"]  = L;
            params["kw"] = kw;

            bridge->call("init", ippl::Comm->rank(), ippl::Comm->size(), Kokkos::device_id(), params);

            inferField();
        }

        void advance() override {
            if (this->stepMethod_m != "LeapFrog")
                throw IpplException("advance()", "Step method not set/Not implemented");

            double dt = this->dt_m;
            auto pc = this->pcontainer_m;
            auto fc = this->fcontainer_m;

            pc->P = pc->P - 0.5 * dt * pc->E;        // kick
            pc->R = pc->R + dt * pc->P;              // drift
            pc->update();                            // migrate + periodic BC

            bool isFirstRepartition = false;
            if (this->loadbalancer_m->balance(this->totalP_m, this->it_m + 1)) {
                auto* mesh = &fc->getRho().get_mesh();
                auto* FL   = &fc->getFL();
                this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
            }

            inferField(); //replaced par2grid, pic solver, then grid2par

            pc->P = pc->P - 0.5 * dt * pc->E;
        }

        void inferField() {
            auto pc = this->pcontainer_m;
            Kokkos::fence();

            py::capsule R_cap = ippl_dlpack::attrib_to_dlpack_vec(
                pc->R, kokkos_dlpack::ReadOnly::Yes, kokkos_dlpack::DLPackVersion::Legacy);

            py::capsule E_cap = ippl_dlpack::attrib_to_dlpack_vec(
                pc->E, kokkos_dlpack::ReadOnly::No, kokkos_dlpack::DLPackVersion::Legacy);

                bridge->call("infer", R_cap, E_cap, pc->getLocalNum());
        }

        void dump() override {
        static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
        IpplTimings::startTimer(dumpDataTimer);

        LandauDampingManager<T,Dim>::dumpLandau();

        IpplTimings::stopTimer(dumpDataTimer);
    }

};
#endif