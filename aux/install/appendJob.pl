#!/usr/bin/perl
# APPENDJOB
# Stephen Gould <sgould@stanford.edu>
#

use Getopt::Std;
use strict;
use Sys::Hostname;
use Fcntl ':flock';

my %opts = ();
getopts("hd:v", \%opts);

if (($opts{h} == 1) || ($#ARGV == -1)) {
    print STDERR "USAGE: appendJob.pl [OPTIONS] <job>\n";
    print STDERR "OPTIONS:\n";
    print STDERR "  -d <dir>     :: directory for jobs files (default: $ENV{HOME})\n";
    print STDERR "  -v           :: verbose\n";
    print STDERR "\n";
    exit(-1);
}

$opts{d} = $ENV{HOME} unless (defined($opts{d}));
my $JOBSQUEUED = "$opts{d}/jobsQueued.txt";

# append job to jobs file
open FILE, ">> $JOBSQUEUED" or die "could not open $JOBSQUEUED\n";
unless (flock FILE, LOCK_EX | LOCK_NB) {
    print STDERR "waiting for lock...\n";
    flock FILE, LOCK_EX or die "could not lock $JOBSQUEUED\n";
}

print FILE join(" ", @ARGV) . "\n";
close FILE;

if (defined($opts{v})) {
    `wc -l $JOBSQUEUED` =~ m/^(\d+)/;
    print STDOUT "$1 jobs queued in $JOBSQUEUED\n";
}
