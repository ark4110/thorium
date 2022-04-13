#pragma once

#include "args.h"

struct params_t
{
  const char* scene_filename = 0;
  const char* task_filename = 0;
  const char* output_filename = 0;
  int normalize_tetra = 0;
};

void print_params(const params_t& params);

struct arguments_t: base_arg_parser_t
{
  arguments_t();

  void scan_args() override;
  void print_usage() override;

  const char* scene_filename = 0;
  const char* task_filename = 0;
  const char* output_filename = 0;
  int normalize_tetra = -1;
};

void apply_args(const arguments_t& args, params_t& params);
