#include "params.h"
#include <cstdio>
#include <cstdlib>

arguments_t::arguments_t()
{
}

void arguments_t::scan_args()
{
  if (int i = find_arg("--obj", false))
    scene_filename = argv[i + 1];
  if (int i = find_arg("--scene", false))
    scene_filename = argv[i + 1];
  if (!scene_filename)
  {
    fprintf(stderr, "Scene not set.\n");
    arg_parse_failed = true;
  }

  if (int i = find_arg("--task", true))
    task_filename = argv[i + 1];

  if (int i = find_arg("--output", false))
    output_filename = argv[i + 1];

  if (int i = find_arg("--normalize-tetra", false))
    normalize_tetra = atoi(argv[i + 1]);
}

void arguments_t::print_usage()
{
  fprintf(stderr, "Usage:\n"
                  "\t--obj scene filename\n"
                  "\t--task task filename\n"
                  "\t--output output filename\n"
                  "\t--normalize-tetra 0 normalize tetrahedrons normals\n"
  );
}

void apply_args(const arguments_t& args, params_t& params)
{
  if (args.scene_filename)
    params.scene_filename = args.scene_filename;
  if (args.task_filename)
    params.task_filename = args.task_filename;
  if (args.output_filename)
    params.output_filename = args.output_filename;
  if (args.normalize_tetra != -1)
    params.normalize_tetra = args.normalize_tetra;
}

void print_params(const params_t& params)
{
  fprintf(stderr, "Simulation params:\n"
                  "\tscene filename: %s\n"
                  "\ttask filename: %s\n"
                  "\toutput: %s\n"
                  "\tnormalize_tetra: %d\n"
                  "\n",
                  params.scene_filename,
                  params.task_filename,
                  params.output_filename ? params.output_filename : "stdout",
                  params.normalize_tetra
  );
}
