# diassep.py

diassep.py takes all .out frequency calculations in the current working directory and splits them into two components and saves the new files in the same directory.

## Usage

### Dependencies
There is only a few dependencies for this code to run:

``` numpy ```

``` cclib ```

``` molml ```

``` xyz_py ```

NOTE: Depending upon the atoms in the systems you may run into trouble - Br is not included in the molml BOND_LENGTHS dictionary so, you will have to navigate to ``` molml/molml/constants.py ``` and modify the file to include Br.

These are likely to all be in an existing environment that the user already has built on the computing resource they are running this on (Grayson Group). 

### Command line usage
Add diassep.py to your scripts folder and make executable.

``` chmod u+x diassep.py ```

To run DIAS separation on all .out/.log files in the current working directory use:

``` python diassep.py ```

This will generate two files for each .out/.log that are named \<filename>_reactant1.gjf and \<filename>_reactant2.gjf in that directory.

You will then have to alter:
- CPU and memory
- Method of choice
- Title (if required)
- Charge and multiplicity

There are a few different optional arguments that can be used with this code. These can be accessed by ``` python diassep.py --help ``` and, are shown below:

```
  -h, --help          show this help message and exit
  -b BREAKBONDS       Comma separated list of numbers of the two atoms that connectivity is to be broken between.
  -i INTRABREAKBONDS  Comma separated list of numbers of the two atoms that start and end the linker - code may be specific for Diels-Alder reaction.
  -a                  Utilise the adjacency matrix approach rather than the distance approach.
  -m                  Provides a pkl file containing a dictionary of all the mapped atoms numbers from the TS to the distorted structures.
  -c COMMONATOMS      Utilises a dictionary of common atoms to separate structures.
  -x                  Perform the DIAS separation for xyz files - requires commonatoms (-c) to be parsed.
  ```


