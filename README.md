# PathFinder

This code helps find the minimum-energy/free-energy path (MEP/MFEP) connecting two endpoints for a given landscape of data points.


# Package requirement
```
pip3 install argparse
pip3 install os
pip3 install re
pip3 install dataclasses 
pip3 install typing
pip3 install numpy
```
# Usage
The help options can be printed with the -h option. The code requires the following input

python PathFinder.py \
  -f energy.dat \
  -end [0.67,15.3] [2.5,21.9] \
  -path [4,4] [18,9] [13.83,17.1] \
  --n-points 80 \
  --n-iter 3000 \
  --dt 0.05 \
  --smooth 0.0 \
  --prefix 2D_Z_MEP

-f energy.dat is the energy matrix on a 2D grid of points. The missing data is interpolated.

-end takes the input of fixed endpoints

-path takes input of waypoints. If not provided, the initial guess is just a linear path between the endpoints.

--n-points are the number of points used to define the path

--n-iter is the attempt number of iterations

--dt is the steps used in the steepest descent minimization

--smooth is used for Gaussian smoothing of data

--prefix is used to dave output with this prefix

NOTE: A demo energy file has been added to the repository, along with the known MFEP and the code's output. Use the following command to run this test case,

python PathFinder.py -f energy.dat -end [-2.5,2.92] [1.23,-1.22] -path [-1.53,1.54] [-1.19,-0.66] --n-points 80 --dt 0.005

# Citations
Please cite the following papers if you are using the code or any segment of code:

```
@article{kulshrestha2022finite,
  title={Finite temperature string method with umbrella sampling using path collective variables: application to secondary structure change in a protein},
  author={Kulshrestha, Avijeet and Punnathanam, Sudeep N and Ayappa, K Ganapathy},
  journal={Soft Matter},
  volume={18},
  number={39},
  pages={7593--7603},
  year={2022},
  publisher={Royal Society of Chemistry}
}
@article{kulshrestha2026transmembrane,
  title={The transmembrane domain regulates the kinetics of the SARS-CoV-2 spike conformational transition},
  author={Kulshrestha, Avijeet and Banerjee, Arkadeep and Lall, Sahil and Gosavi, Shachi},
  journal={bioRxiv},
  pages={2026--04},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
``` 

Best Regards.  
Avijeet Kulshrestha



