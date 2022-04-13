#pragma once

struct base_arg_parser_t
{
    virtual ~base_arg_parser_t() = default;

    int parse_args(int _argc, char **_argv);
    virtual void scan_args() = 0;

    virtual void print_usage() = 0;

protected:
    int find_arg(const char *name, bool required = false);

protected:
    int argc;
    char **argv;
    bool arg_parse_failed;
};
