#include "optionparser.h"

struct Arg: public option::Arg
{
    static void printError(const char* msg1, const option::Option& opt, const char* msg2)
    {
        fprintf(stderr, "%s", msg1);
        fwrite(opt.name, opt.namelen, 1, stderr);
        fprintf(stderr, "%s", msg2);
    }

    static option::ArgStatus Unknown(const option::Option& option, bool msg)
    {
        if (msg) printError("Unknown option '", option, "'\n");
        return option::ARG_ILLEGAL;
    }

    static option::ArgStatus Required(const option::Option& option, bool msg)
    {
        if (option.arg != 0)
            return option::ARG_OK;

        if (msg) printError("Option '", option, "' requires an argument\n");
        return option::ARG_ILLEGAL;
    }

    static option::ArgStatus NonEmpty(const option::Option& option, bool msg)
    {
        if (option.arg != 0 && option.arg[0] != 0)
            return option::ARG_OK;

        if (msg) printError("Option '", option, "' requires a non-empty argument\n");
        return option::ARG_ILLEGAL;
    }

    static option::ArgStatus Numeric(const option::Option& option, bool msg)
    {
        char* endptr = 0;
        if (option.arg != 0 && strtol(option.arg, &endptr, 10)){};
        if (endptr != option.arg && *endptr == 0)
            return option::ARG_OK;

        if (msg) printError("Option '", option, "' requires a numeric argument\n");
        return option::ARG_ILLEGAL;
    }
};

enum  optionIndex {
    UNKNOWN, HELP, THREAD_COUNT,
    RANDOM_SEED, INPUT_LIST, LABEL_LIST,
    VOI_PER_LABEL, OUTPUT_PATH, DIAMETER,
};

const option::Descriptor usage[] =
{
    {UNKNOWN, 0,"" , ""    ,option::Arg::None, "USAGE: DICOMProcessor [options]\n\n" "Options:" },
    {HELP, 0,"" , "help",option::Arg::None, "  --help  \tPrint usage and exit." },

    {INPUT_LIST, 0, "i", "input", Arg::NonEmpty, "  --input, -i  \tSpecifies one directory to process (Can specify more than one)." },
    {LABEL_LIST, 0, "l", "label", Arg::NonEmpty, "  --label, -l  \tSpecifies one label to use (Can specify more than one)." },
    {OUTPUT_PATH, 0, "o", "output-path", Arg::NonEmpty, "  --output-pat, -o  \tOutput path." },

    {THREAD_COUNT, 0, "t", "thread-count", Arg::Numeric, "  --thread-count, -t  \tNumber of thread (default : 3)." },
    {RANDOM_SEED, 0, "r", "random-seed", Arg::Numeric, "  --random-seed, -r  \tRandom Seed (default : 1)." },
    {DIAMETER, 0, "d", "diameter", Arg::Numeric, "  --diameter, -d  \tVOI Diameter, must be odd (default : 25)." },
    {VOI_PER_LABEL, 0, "v", "voi-per-label", Arg::Numeric, "  --voi-per-label, -v  \tNumber of voi extracted for each label (default : 5)." },

    {0,0,0,0,0,0}
};
