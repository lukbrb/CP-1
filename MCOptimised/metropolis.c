#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_MOVES 500000
#define DENSITY 0.0155  // Number of Krypton atoms per cubic angstrom

// dimensions of the box on each axis, in Angstrom
#define NDIM 3
#define LX 15
#define LY 15
#define LZ 15
#define BOX_VOLUME (LX * LY * LZ)
#define MAG_MOVE 0.3 // magnitude of the displacement

#define SIGMA 3.624   // vdw radius [angstrom]
#define EPSILON 0.317 // vdw energy [kcal/mol]
#define TEMPERATURE 115.79  // Melting point of Krypton

// prototypes of the functions

double rand(double start, double end);

double make_periodic(double r, double boundary);

double wrap_into_box(double r, double boundary);

double get_energy(double *coords, int n_atoms, double *box_size);

void copy_coordinates(double **from_array, double **new_array);

void print_pdb(double **coords, int n_atoms, int num_move);

int main(int argc, char *argv[])
{
    const int N_ATOMS = BOX_VOLUME * DENSITY;
    const double kB = 1.987206504191549E-003;  // kcal mol-1 K-1
    const double kT = kB * TEMPERATURE;
    const float BOX_SIZE[NDIM] = {LX, LY, LZ};

    int n_accept = 0;
    int n_reject = 0;
    
    double *coords = NULL;
    double *old_coords=NULL;

    coords = (double *)malloc((N_ATOMS * NDIM) * sizeof(double));
    old_coords = (double *)malloc(N_ATOMS * sizeof(double));

    for (int i = 0; i < N_ATOMS; i++)
    {
        coords[i][0] = rand(0, LX);
        coords[i][1] = rand(0, LY);
        coords[i][2] = rand(0, LZ);
    }

    // print the initial PDB file
    print_pdb(coords, N_ATOMS, 0);

    for (int move = 1; move <= num_moves; move++)
    {
        double old_energy = get_energy(coords, N_ATOMS, BOX_SIZE);

        int atom = int(rand(0, n_atoms));

        copy_coordinates(coords, old_coords);

        double dx = rand(-MAG_MOVE, MAG_MOVE);
        double dy = rand(-MAG_MOVE, MAG_MOVE);
        double dz = rand(-MAG_MOVE, MAG_MOVE);

        coords[atom][0] += dx;
        coords[atom][1] += dy;
        coords[atom][2] += dz;

        double new_energy = get_energy(coords, N_ATOMS, BOX_SIZE);
        double total_energy = 0;


        if (new_energy <= old_energy)
        {
            n_accept += 1;
            total_energy = new_energy;
        }

        else
        {
            double proba = exp(-(new_energy - old_energy)/kT);

            if (proba >= rand(0.0, 1.0))
            {
                n_accept += 1;
                total_energy = new_energy;
            }

            else
            {
                n_reject += 1;
                copy_coordinates(old_coords, coords);
                total_energy = old_energy;
            }
        }   

        if (move % 1000 == 0)
        {
            printf("%d %f  %d  %d\n", move, total_energy, naccept, nreject);
        }

        // print the coordinates every 10000 moves
        if (move % 10000 == 0)
        {
            print_pdb(coords, n_atoms, move);
        }
    }

    retrun 0;
}



