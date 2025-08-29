cmake_minimum_required(VERSION 3.12...3.28)
project(HeatEquation2D LANGUAGES CXX)

find_package(MPI REQUIRED)
find_package(VTK REQUIRED COMPONENTS CommonCore CommonDataModel IOXML)

add_executable(heat_equation 2dHeatEquationV3.cpp)
target_link_libraries(heat_equation PRIVATE MPI::MPI_CXX VTK::CommonCore VTK::CommonDataModel VTK::IOXML)
target_compile_features(heat_equation PRIVATE cxx_std_17)