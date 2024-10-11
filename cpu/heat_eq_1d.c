#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <assert.h>

#if defined(_WIN32) || defined(_WIN64)
#include <malloc.h>

double *allocate_aligned_memory (size_t size, size_t alignment)
{
  return (double *) _aligned_malloc (size * sizeof (double), alignment);
}

void free_aligned_memory (void *ptr)
{
  _aligned_free (ptr);
}
#else
double *allocate_aligned_memory (size_t size, size_t alignment)
{
  double *ptr;
  if (posix_memalign ((void **) &ptr, alignment, size * sizeof (double)) != 0)
  {
    return NULL;
  }
  return ptr;
}
void free_aligned_memory (void *ptr)
{
  free (ptr);
}
#endif

// Structure to hold simulation parameters
typedef struct
{
  double alpha;	     // Thermal diffusivity
  double dt;	     // Time step size
  double dx;	     // Spatial step size
  size_t n;	     // Number of spatial points
  size_t time_steps; // Total number of time steps
} HeatEquationParams;

/**
 * Solves the 1d heat eqn using explicit finite difference method.
 *
 * @param u           Pointer to the initial temperature array.
 * @param params      Pointer to the HeatEquationParams structure containing
 * simulation parameters.
 * @return            0 success, -1 failure.
 */

int solve_heat_equation (double *u, const HeatEquationParams *params)
{
  if (u == NULL || params == NULL || params->n <= 2)
  {
    fprintf (stderr, "Error: Invalid input parameters.\n");
    return -1;
  }

  // Calculate stability coefficient
  const double coeff = params->alpha * params->dt / (params->dx * params->dx);
  const size_t n = params->n;
  const size_t time_steps = params->time_steps;

  // Check stability condition
  if (coeff > 0.5)
  {
    fprintf (stderr,
         "Error: Stability condition violated (coeff = %f > 0.5).\n",
         coeff);
    return -1;
  }

  // Allocate memory for new temperature array with 64-byte alignment
  double *u_new = allocate_aligned_memory (n, 64);
  if (u_new == NULL)
  {
    fprintf (stderr, "Error: Memory allocation failed for u_new.\n");
    return -1;
  }

  // Initialize u_new with initial temperatures
  memcpy (u_new, u, n * sizeof (double));

  // Time-stepping loop
  for (size_t t = 0; t < time_steps; t++)
  {
// Update temperatures in parallel
#pragma omp parallel for schedule(static)
    for (size_t i = 1; i < n - 1; i++)
      u_new[i] = u[i] + coeff * (u[i + 1] - 2.0 * u[i] + u[i - 1]);

// Apply boundary conditions
#pragma omp single
    {
      u_new[0] = 0.0;
      u_new[n - 1] = 0.0;
    }

    // Swap arrays for next step
    double *temp = u;
    u = u_new;
    u_new = temp;
  }

  // Copy final results back to u if needed
  if (u != u_new)
    memcpy (u, u_new, n * sizeof (double));

  free_aligned_memory (u_new);

  return 0;
}

int main (void)
{
  HeatEquationParams params = {
    .alpha = 1.0,     // Thermal diffusivity
    .dt = 0.004,      // Time step size (adjusted to ensure coeff <= 0.5)
    .dx = 0.1,	      // Spatial step size
    .n = 10000,	      // Number of spatial points
    .time_steps = 100 // Total number of time steps
  };

  double *u = allocate_aligned_memory (params.n, 64);
  if (u == NULL)
  {
    fprintf (stderr, "Error: Memory allocation failed for u.\n");
    return -1;
  }

  memset (u, 0, params.n * sizeof (double));

  // Set initial heat pulse in the center
  u[params.n / 2] = 100.0;

  // Solve the heat equation
  if (solve_heat_equation (u, &params) != 0)
  {
    fprintf (stderr, "Error: Heat equation solver failed.\n");
    free_aligned_memory (u);
    return -1;
  }

  // for (size_t i = 0; i < params.n; i++) {
  //   printf ("u[%zu] = %f\n", i, u[i]);
  // }

  free_aligned_memory (u);

  return 0;
}