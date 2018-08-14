#!/usr/bin/perl
# MANAGEJOBS
# Stephen Gould <sgould@stanford.edu>
#

use Getopt::Std;
use strict;
use Sys::Hostname;
use Fcntl ':flock';

my %opts = ();
getopts("ahd:pvx:", \%opts);

if (($opts{h} == 1) || ($#ARGV != -1)) {
    print STDERR "USAGE: manageJobs.pl [OPTIONS]\n";
    print STDERR "OPTIONS:\n";
    print STDERR "  -a           :: re-append deleted jobs\n";
    print STDERR "  -d <dir>     :: directory for jobs files (default: $ENV{HOME})\n";
    print STDERR "  -p           :: purge all jobs\n";
    print STDERR "  -v           :: verbose\n";
    print STDERR "  -x <host>    :: delete running jobs on host\n";
    print STDERR "\n";
    exit(-1);
}

$opts{d} = $ENV{HOME} unless (defined($opts{d}));
my $JOBSRUNNING = "$opts{d}/jobsRunning.txt";

# keep track of deleted jobs
my $counter = 0;
my @deletedJobs = ();

# purge all jobe
if (defined($opts{p})) {
    open FILE, "+< $JOBSRUNNING" or die "could not open $JOBSRUNNING\n";
    flock FILE, LOCK_EX or die "could not lock $JOBSRUNNING\n";
    @deletedJobs = <FILE>;
    $counter = $#deletedJobs + 1;

    seek FILE, 0, 0;
    truncate FILE, tell FILE;
    close FILE;

    for (my $i = 0; $i <= $#deletedJobs; $i++) {
        $deletedJobs[$i] =~ s/^.*\t.*\t//;
        chomp $deletedJobs[$i];
    }
}

# delete matching jobs
if (defined($opts{x})) {
    open FILE, "+< $JOBSRUNNING" or die "could not open $JOBSRUNNING\n";
    flock FILE, LOCK_EX or die "could not lock $JOBSRUNNING\n";
    my @allRunningJobs = <FILE>;
    seek FILE, 0, 0;
    for (my $i = 0; $i <= $#allRunningJobs; $i++) {
        chomp $allRunningJobs[$i];
        if ($allRunningJobs[$i] !~ m/$opts{x}/) {
            print FILE "$allRunningJobs[$i]\n";
        } else {
            $allRunningJobs[$i] =~ m/([^\t]+)$/;
            push @deletedJobs, $1;
            $counter += 1;
        }
    }
    truncate FILE, tell FILE;
    close FILE;
}

# delete old jobs
# TODO

# show deleted jobs
print "...deleted $counter jobs from $JOBSRUNNING\n";

# append deleted jobs
if (defined($opts{a})) {
    for (my $i = 0; $i <= $#deletedJobs; $i++) {
        $deletedJobs[$i] =~ s/([^a-zA-Z0-9_ :.\-\/])/\\\1/g;
        print "$deletedJobs[$i]\n";
        `./appendJob.pl $deletedJobs[$i]`;
    }
}
