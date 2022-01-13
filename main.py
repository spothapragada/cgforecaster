import logging
import argparse

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="A simple script to test logging and argparse")

    parser.add_argument("-v", "--verbose", action="store_true", required=True, help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", required=True, help="increase output verbosity")
    args = parser.parse_args()

    if args.verbose:
        logging.info("Verbose mode is on")
    else:
        logging.info("Verbose mode is off")

    if args.debug:
        logging.info("Debug-name mode is on and is {}".format(args['debug']))

    logging.info("This is a test")



