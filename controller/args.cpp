#include "args.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>

int ArgPos(const char *str, int argc, char **argv)
{
  int a;
  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        fprintf(stderr, "Argument missing for %s\n", str);
        exit(-1);
      }
      return a;
    }
  return 0;
}

int base_arg_parser_t::parse_args(int _argc, char **_argv)
{
  if (_argc == 1) {
    print_usage();
    return -1;
  }

  argc = _argc;
  argv = _argv;

  arg_parse_failed = false;
  scan_args();
  if (arg_parse_failed) {
    print_usage();
    return -1;
  }

  return 0;
}

int base_arg_parser_t::find_arg(const char *name, bool required)
{
  int i = ArgPos(name, argc, argv);
  if (required && !i) {
    fprintf(stderr, "Required arg %s not found\n", name);
    arg_parse_failed = true;
  }
  return i;
}

