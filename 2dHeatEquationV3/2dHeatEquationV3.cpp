#include <vector>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>
#include <vtkStructuredGrid.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkPointData.h>
#include <mpi.h>
#include <string>
#include <cmath>
#include <filesystem>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Physical parameters
    const double k = 1.172e-5; // thermal diffusivity (steel, 1% carbon)
    const double Lx = 0.1;     // length
    const double Ly = 0.1;     // width

    // Numerical parameters
    const int nx = 40;         // number of points in x direction
    const int ny = 40;         // number of points in y direction
    const double dt = 0.1;     // time step
    const double tf = 10.0;    // final time
    const int nt = static_cast<int>(tf / dt); // number of time steps

    // Boundary conditions
    const double T0 = 1.0;     // internal field
    const double T1 = 0.0;     // bottom boundary
    const double T2 = 0.0;     // top boundary
    const double T3 = 0.0;     // left boundary
    const double T4 = 0.0;     // right boundary

    // Compute cell lengths
    const double dx = Lx / nx;
    const double dy = Ly / ny;

    // Courant numbers
    double r1 = k * dt / (dx * dx);
    double r2 = k * dt / (dy * dy);
    if (rank == 0) {
        if (r1 > 0.5 || r2 > 0.5) {
            throw std::runtime_error("Unstable Solution! r1=" + std::to_string(r1) + ", r2=" + std::to_string(r2));
        }
    }

    // Timing variables
    double start_time, end_time, compute_time = 0.0, comm_time = 0.0;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    // Domain decomposition: divide ny rows among processes
    int local_ny = ny / size;
    int remainder = ny % size;
    if (rank < remainder) local_ny++;
    int local_start_y = 0;
    for (int p = 0; p < rank; p++) {
        local_start_y += (ny / size) + (p < remainder ? 1 : 0);
    }
    int local_end_y = local_start_y + local_ny;

    // Local T slice for each process (no halos, boundaries set globally)
    std::vector<double> local_T_slice(nx * local_ny, 0.0);

    // Full T grid on rank 0
    std::vector<std::vector<double>> full_T;
    if (rank == 0) {
        full_T = std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0)); 
    }

    // VTK setup on rank 0
    vtkSmartPointer<vtkStructuredGrid> structuredGrid;
    vtkSmartPointer<vtkPoints> points;
    vtkSmartPointer<vtkDoubleArray> temperature;
    vtkSmartPointer<vtkXMLStructuredGridWriter> writer;
    std::ofstream pvdFile;
    if (rank == 0) {
        // Create result directory
        std::filesystem::create_directory("result");

        structuredGrid = vtkSmartPointer<vtkStructuredGrid>::New();
        structuredGrid->SetDimensions(nx, ny, 1);

        points = vtkSmartPointer<vtkPoints>::New();
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                points->InsertNextPoint(i * dx, j * dy, 0.0);
            }
        }
        structuredGrid->SetPoints(points);

        temperature = vtkSmartPointer<vtkDoubleArray>::New();
        temperature->SetName("Temperature");

        writer = vtkSmartPointer<vtkXMLStructuredGridWriter>::New();

        // Create .pvd file in result directory
        pvdFile.open("result/timeseries.pvd");
        pvdFile << "<?xml version=\"1.0\"?>\n"
            << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
            << "  <Collection>\n";
    }

    // Gather and output initial condition (t=0)
    std::vector<int> counts(size), displs(size);
    for (int p = 0; p < size; p++) {
        counts[p] = (ny / size + (p < ny % size ? 1 : 0)) * nx;
        displs[p] = (p == 0 ? 0 : displs[p - 1] + counts[p - 1]);
    }

    // Rank 0 sets initial conditions
    if (rank == 0) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                full_T[i][j] = T0; 
            }
        }
        // Set boundaries globally on rank 0
        for (int i = 0; i < nx; ++i) {
            full_T[i][0] = T1; // bottom
            full_T[i][ny - 1] = T2; // top
        }
        for (int j = 0; j < ny; ++j) {
            full_T[0][j] = T3; // left
            full_T[nx - 1][j] = T4; // right
        }

        // Output initial condition
        temperature->Reset();
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                temperature->InsertNextValue(full_T[i][j]); 
            }
        }
        structuredGrid->GetPointData()->SetScalars(temperature);
        writer->SetFileName("result/output_step_0.vts");
        writer->SetInputData(structuredGrid);
        writer->Write();
        pvdFile << "    <DataSet timestep=\"0.0\" part=\"0\" file=\"output_step_0.vts\"/>\n";
    }

    // Broadcast initial full T to all processes
    std::vector<double> full_slice(nx * ny);
    if (rank == 0) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                full_slice[j * nx + i] = full_T[i][j]; 
            }
        }
    }
    double comm_start = MPI_Wtime();
    MPI_Bcast(full_slice.data(), nx * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter to local slices
    MPI_Scatterv(full_slice.data(), counts.data(), displs.data(), MPI_DOUBLE,
        local_T_slice.data(), nx * local_ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    comm_time += MPI_Wtime() - comm_start;

    // Main time loop
    for (int t = 0; t < nt - 1; ++t) {
        // Create local slice for previous time step
        std::vector<double> local_slice_prev(nx * local_ny);
        for (int local_j = 0; local_j < local_ny; ++local_j) {
            for (int i = 0; i < nx; ++i) {
                local_slice_prev[local_j * nx + i] = local_T_slice[local_j * nx + i];
            }
        }

        // Solve heat equation locally (internal points only)
        double compute_start = MPI_Wtime();
        for (int i = 1; i < nx - 1; ++i) {
            for (int local_j = 0; local_j < local_ny; ++local_j) {
                if (local_start_y + local_j > 0 && local_start_y + local_j < ny - 1) {
                    double d2dx2 = (local_slice_prev[local_j * nx + i + 1] - 2 * local_slice_prev[local_j * nx + i] + local_slice_prev[local_j * nx + i - 1]) / (dx * dx);
                    double d2dy2 = (full_slice[(local_start_y + local_j + 1) * nx + i] - 2 * full_slice[(local_start_y + local_j) * nx + i] + full_slice[(local_start_y + local_j - 1) * nx + i]) / (dy * dy);
                    local_T_slice[local_j * nx + i] = k * dt * (d2dx2 + d2dy2) + local_slice_prev[local_j * nx + i];
                    if (std::isnan(local_T_slice[local_j * nx + i]) || std::isinf(local_T_slice[local_j * nx + i])) {
                        throw std::runtime_error("Invalid value detected at t=" + std::to_string(t) + ", i=" + std::to_string(i) + ", local_j=" + std::to_string(local_j));
                    }
                }
            }
        }
        compute_time += MPI_Wtime() - compute_start;

        // Gather updated local slices to rank 0
        comm_start = MPI_Wtime();
        MPI_Gatherv(local_T_slice.data(), nx * local_ny, MPI_DOUBLE,
            full_slice.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        comm_time += MPI_Wtime() - comm_start;

        // Rank 0 updates full T
        if (rank == 0) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    full_T[i][j] = full_slice[j * nx + i];
                }
            }

            // Output t+1
            temperature->Reset();
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    temperature->InsertNextValue(full_T[i][j]);
                }
            }
            structuredGrid->GetPointData()->SetScalars(temperature);
            std::string filename = "result/output_step_" + std::to_string(t + 1) + ".vts";
            writer->SetFileName(filename.c_str());
            writer->SetInputData(structuredGrid);
            writer->Write();
            pvdFile << "    <DataSet timestep=\"" << (t + 1) * dt << "\" part=\"0\" file=\"output_step_" << (t + 1) << ".vts\"/>\n";
        }

        // Broadcast updated full T for next iteration
        comm_start = MPI_Wtime();
        if (rank == 0) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    full_slice[j * nx + i] = full_T[i][j]; 
                }
            }
        }
        MPI_Bcast(full_slice.data(), nx * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        comm_time += MPI_Wtime() - comm_start;

        // Scatter to local slices for next iteration
        comm_start = MPI_Wtime();
        MPI_Scatterv(full_slice.data(), counts.data(), displs.data(), MPI_DOUBLE,
            local_T_slice.data(), nx * local_ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        comm_time += MPI_Wtime() - comm_start;
    }

    // Close .pvd file and report timing on rank 0
    if (rank == 0) {
        pvdFile << "  </Collection>\n"
            << "</VTKFile>\n";
        pvdFile.close();
        end_time = MPI_Wtime();
        std::cout << "Simulation completed. Output written to result/timeseries.pvd" << std::endl;
        std::cout << "Total time: " << end_time - start_time << " seconds" << std::endl;
        std::cout << "Compute time: " << compute_time << " seconds" << std::endl;
        std::cout << "Communication time: " << comm_time << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}