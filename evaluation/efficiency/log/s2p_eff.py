# coding: utf-8
import sys
import numpy
import math
import argparse

def find_nearest_index(value, array):
    '''
    find the index of nearest value from number array
    ex: value = 100, array = [10, 200, 300] -> 200
    '''
    dist_list = [abs(v - value) for v in array]
    return dist_list.index(min(dist_list))

def parse_number_with_unit(num_str):
    '''
    parse number string which includes unit string such as 'G', 'M', 'K', 'm'
    '''
    unit_list = 'T G M K k m'.split()
    unit_index_list = [12, 9, 6, 3, 3, -3]
    if not isinstance(num_str, str):
      return None
    unit_value = 1.
    base_num_str = num_str
    if num_str[-1] in unit_list:
      unit_value = 10 ** unit_index_list[unit_list.index(num_str[-1])]
      base_num_str = base_num_str[:-1]
    try:
      return float(base_num_str) * unit_value
    except ValueError:
      return None

'''
Copyright (c) 2010, Alexander Arsenovic
All rights reserved.

Copyright (c) 2017, scikit-rf Developers
All rights reserved.
'''
def get_fid(file, *args, **kwargs):
    '''
    Returns a file object, given a filename or file object
    Useful when you want to allow the arguments of a function to
    be either files or filenames
    Parameters
    -------------
    file : str/unicode or file-object
        file to open
    \*args, \*\*kwargs : arguments and keyword arguments to `open()`
    '''
    if isinstance(file, str):
        return open(file, *args, **kwargs)
    else:
        return file

class Touchstone:
    """
    class to read touchstone s-parameter files
    The reference for writing this class is the draft of the
    Touchstone(R) File Format Specification Rev 2.0 [#]_ and
    Touchstone(R) File Format Specification Version 2.0 [##]_
    .. [#] https://ibis.org/interconnect_wip/touchstone_spec2_draft.pdf
    .. [##] https://ibis.org/touchstone_ver2.0/touchstone_ver2_0.pdf
    """
    def __init__(self, file):
        """
        constructor
        Parameters
        -------------
        file : str or file-object
            touchstone file to load
        Examples
        ---------
        From filename
        >>> t = rf.Touchstone('network.s2p')
        From file-object
        >>> file = open('network.s2p')
        >>> t = rf.Touchstone(file)
        """
        fid = get_fid(file)
        filename = fid.name
        ## file name of the touchstone data file
        self.filename = filename

        ## file format version.
        # Defined by default to 1.0, since version number can be omitted in V1.0 format
        self.version = '1.0'
        ## comments in the file header
        self.comments = None
        ## unit of the frequency (Hz, kHz, MHz, GHz)
        self.frequency_unit = None
        ## number of frequency points
        self.frequency_nb = None
        ## s-parameter type (S,Y,Z,G,H)
        self.parameter = None
        ## s-parameter format (MA, DB, RI)
        self.format = None
        ## reference resistance, global setup
        self.resistance = None
        ## reference impedance for each s-parameter
        self.reference = None

        ## numpy array of original s-parameter data
        self.sparameters = None
        ## numpy array of original noise data
        self.noise = None

        ## kind of s-parameter data (s1p, s2p, s3p, s4p)
        self.rank = None
        ## Store port names in a list if they exist in the file
        self.port_names = None

        self.comment_variables=None
        self.load_file(fid)

    def load_file(self, fid):
        """
        Load the touchstone file into the interal data structures
        """

        filename=self.filename

        # Check the filename extension.
        # Should be .sNp for Touchstone format V1.0, and .ts for V2
        extension = filename.split('.')[-1].lower()

        if (extension[0] == 's') and (extension[-1] == 'p'): # sNp
            # check if N is a correct unmber
            try:
                self.rank = int(extension[1:-1])
            except (ValueError):
                raise (ValueError("filename does not have a s-parameter extension. It has  [%s] instead. please, correct the extension to of form: 'sNp', where N is any integer." %(extension)))
        elif extension == 'ts':
            pass
        else:
            raise Exception('Filename does not have the expected Touchstone extension (.sNp or .ts)')

        linenr = 0
        values = []
        while (1):
            linenr +=1
            line = fid.readline()
            if not type(line) == str:
                line = line.decode("ascii")  # for python3 zipfile compatibility
            if not line:
                break

            # store comments if they precede the option line
            line = line.split('!',1)
            if len(line) == 2:
                if not self.parameter:
                    if self.comments == None:
                        self.comments = ''
                    self.comments = self.comments + line[1]
                elif line[1].startswith(' Port['):
                    try:
                        port_string, name = line[1].split('=', 1) #throws ValueError on unpack
                        name = name.strip()
                        garbage, index = port_string.strip().split('[', 1) #throws ValueError on unpack
                        index = int(index.rstrip(']')) #throws ValueError on not int-able
                        if index > self.rank or index <= 0:
                            print("Port name {0} provided for port number {1} but that's out of range for a file with extension s{2}p".format(name, index, self.rank))
                        else:
                            if self.port_names is None: #Initialize the array at the last minute
                                self.port_names = [''] * self.rank
                            self.port_names[index - 1] = name
                    except ValueError as e:
                        print("Error extracting port names from line: {0}".format(line))

            # remove the comment (if any) so rest of line can be processed.
            # touchstone files are case-insensitive
            line = line[0].strip().lower()

            # skip the line if there was nothing except comments
            if len(line) == 0:
                continue

            # grab the [version] string
            if line[:9] == '[version]':
                self.version = line.split()[1]
                continue

            # grab the [reference] string
            if line[:11] == '[reference]':
                # The reference impedances can be span after the keyword
                # or on the following line
                self.reference = [ float(r) for r in line.split()[2:] ]
                if not self.reference:
                    line = fid.readline()
                    self.reference = [ float(r) for r in line.split()]
                continue

            # grab the [Number of Ports] string
            if line[:17] == '[number of ports]':
                self.rank = int(line.split()[-1])
                continue

            # grab the [Number of Frequencies] string
            if line[:23] == '[number of frequencies]':
                self.frequency_nb = line.split()[-1]
                continue

            # skip the [Network Data] keyword
            if line[:14] == '[network data]':
                continue

            # skip the [End] keyword
            if line[:5] == '[end]':
                continue

            # the option line
            if line[0] == '#':
                toks = line[1:].strip().split()
                # fill the option line with the missing defaults
                toks.extend(['ghz', 's', 'ma', 'r', '50'][len(toks):])
                self.frequency_unit = toks[0]
                self.parameter = toks[1]
                self.format = toks[2]
                self.resistance = toks[4]
                if self.frequency_unit not in ['hz', 'khz', 'mhz', 'ghz']:
                    print('ERROR: illegal frequency_unit [%s]',  self.frequency_unit)
                    # TODO: Raise
                if self.parameter not in 'syzgh':
                    print('ERROR: illegal parameter value [%s]', self.parameter)
                    # TODO: Raise
                if self.format not in ['ma', 'db', 'ri']:
                    print('ERROR: illegal format value [%s]', self.format)
                    # TODO: Raise

                continue

            # collect all values without taking care of there meaning
            # we're seperating them later
            values.extend([ float(v) for v in line.split() ])

        # let's do some post-processing to the read values
        # for s2p parameters there may be noise parameters in the value list
        values = numpy.asarray(values)
        if self.rank == 2:
            # the first frequency value that is smaller than the last one is the
            # indicator for the start of the noise section
            # each set of the s-parameter section is 9 values long
            pos = numpy.where(numpy.sign(numpy.diff(values[::9])) == -1)
            if len(pos[0]) != 0:
                # we have noise data in the values
                pos = pos[0][0] + 1   # add 1 because diff reduced it by 1
                noise_values = values[pos*9:]
                values = values[:pos*9]
                self.noise = noise_values.reshape((-1,5))

        if len(values)%(1+2*(self.rank)**2) != 0 :
            # incomplete data line / matrix found
            raise AssertionError

        # reshape the values to match the rank
        self.sparameters = values.reshape((-1, 1 + 2*self.rank**2))
        # multiplier from the frequency unit
        self.frequency_mult = {'hz':1.0, 'khz':1e3,
                               'mhz':1e6, 'ghz':1e9}.get(self.frequency_unit)
        # set the reference to the resistance value if no [reference] is provided
        if not self.reference:
            self.reference = [self.resistance] * self.rank

    def get_comments(self, ignored_comments = ['Created with skrf']):
        """
        Returns the comments which appear anywhere in the file.  Comment lines
        containing ignored comments are removed.  By default these are comments
        which contain special meaning withing skrf and are not user comments.
        """
        processed_comments = ''
        if self.comments is None:
            self.comments = ''
        for comment_line in self.comments.split('\n'):
            for ignored_comment in ignored_comments:
                if ignored_comment in comment_line:
                        comment_line = None
            if comment_line:
                processed_comments = processed_comments + comment_line + '\n'
        return processed_comments

    def get_comment_variables(self):
        '''
        convert hfss variable comments to a dict of vars:(numbers,units)
        '''
        comments = self.comments
        p1 = re.compile('\w* = \w*')
        p2 = re.compile('\s*(\d*)\s*(\w*)')
        var_dict = {}
        for k in re.findall(p1, comments):
            var, value = k.split('=')
            var=var.rstrip()
            try:
                var_dict[var] = p2.match(value).groups()
            except:
                pass
        return var_dict

    def get_format(self, format="ri"):
        """
        returns the file format string used for the given format.
        This is useful to get some information.
        """
        if format == 'orig':
            frequency = self.frequency_unit
            format = self.format
        else:
            frequency = 'hz'
        return "%s %s %s r %s" %(frequency, self.parameter,
                                 format, self.resistance)


    def get_sparameter_names(self, format="ri"):
        """
        generate a list of column names for the s-parameter data
        The names are different for each format.
        posible format parameters:
          ri, ma, db, orig  (where orig refers to one of the three others)
        returns a list of strings.
        """
        names = ['frequency']
        if format == 'orig':
            format = self.format
        ext1, ext2 = {'ri':('R','I'),'ma':('M','A'), 'db':('DB','A')}.get(format)
        for r1 in xrange(self.rank):
            for r2 in xrange(self.rank):
                names.append("S%i%i%s"%(r1+1,r2+1,ext1))
                names.append("S%i%i%s"%(r1+1,r2+1,ext2))
        return names

    def get_sparameter_data(self, format='ri'):
        """
        get the data of the s-parameter with the given format.
        supported formats are:
          orig:  unmodified s-parameter data
          ri:    data in real/imaginary
          ma:    data in magnitude and angle (degree)
          db:    data in log magnitute and angle (degree)
        Returns a list of numpy.arrays
        """
        ret = {}
        if format == 'orig':
            values = self.sparameters
        else:
            values = self.sparameters.copy()
            # use frequency in hz unit
            values[:,0] = values[:,0]*self.frequency_mult
            if (self.format == 'db') and (format == 'ma'):
                values[:,1::2] = 10**(values[:,1::2]/20.0)
            elif (self.format == 'db') and (format == 'ri'):
                v_complex = ((10**values[:,1::2]/20.0)
                             * numpy.exp(1j*numpy.pi/180 * values[:,2::2]))
                values[:,1::2] = numpy.real(v_complex)
                values[:,2::2] = numpy.imag(v_complex)
            elif (self.format == 'ma') and (format == 'db'):
                values[:,1::2] = 20*numpy.log10(values[:,1::2])
            elif (self.format == 'ma') and (format == 'ri'):
                v_complex = (values[:,1::2] * numpy.exp(1j*numpy.pi/180 * values[:,2::2]))
                values[:,1::2] = numpy.real(v_complex)
                values[:,2::2] = numpy.imag(v_complex)
            elif (self.format == 'ri') and (format == 'ma'):
                v_complex = numpy.absolute(values[:,1::2] + 1j* self.sparameters[:,2::2])
                values[:,1::2] = numpy.absolute(v_complex)
                values[:,2::2] = numpy.angle(v_complex)*(180/numpy.pi)
            elif (self.format == 'ri') and (format == 'db'):
                v_complex = numpy.absolute(values[:,1::2] + 1j* self.sparameters[:,2::2])
                values[:,1::2] = 20*numpy.log10(numpy.absolute(v_complex))
                values[:,2::2] = numpy.angle(v_complex)*(180/numpy.pi)

        for i,n in enumerate(self.get_sparameter_names(format=format)):
            ret[n] = values[:,i]
        return ret

    def get_sparameter_arrays(self):
        """
        Returns the s-parameters as a tuple of arrays, where the first element is
        the frequency vector (in Hz) and the s-parameters are a 3d numpy array.
        The values of the s-parameters are complex number.
        usage:
          f,a = self.sgetparameter_arrays()
          s11 = a[:,0,0]
        """
        v = self.sparameters

        if self.format == 'ri':
            v_complex = v[:,1::2] + 1j* v[:,2::2]
        elif self.format == 'ma':
            v_complex = (v[:,1::2] * numpy.exp(1j*numpy.pi/180 * v[:,2::2]))
        elif self.format == 'db':
            v_complex = ((10**(v[:,1::2]/20.0)) * numpy.exp(1j*numpy.pi/180 * v[:,2::2]))

        if self.rank == 2 :
            # this return is tricky; it handles the way touchtone lines are
            # in case of rank==2: order is s11,s21,s12,s22
            return (v[:,0] * self.frequency_mult,
                    numpy.transpose(v_complex.reshape((-1, self.rank, self.rank)),axes=(0,2,1)))
        else:
            return (v[:,0] * self.frequency_mult,
                    v_complex.reshape((-1, self.rank, self.rank)))

    def get_noise_names(self):
        """
        TODO: NIY
        """
        TBD = 1


    def get_noise_data(self):
        """
        TODO: NIY
        """
        TBD = 1
        noise_frequency = noise_values[:,0]
        noise_minimum_figure = noise_values[:,1]
        noise_source_reflection = noise_values[:,2]
        noise_source_phase = noise_values[:,3]
        noise_normalized_resistance = noise_values[:,4]

    def is_from_hfss(self):
        '''
        Check if the Touchstone file has been produced by HFSS

        Returns
        ------------
        status : boolean
            True if the Touchstone file has been produced by HFSS
            False otherwise
        '''
        status = False
        if 'exported from hfss' in str.lower(self.comments):
            status = True
        return status

    def get_gamma_z0(self):
        '''
        Extracts Z0 and Gamma comments from touchstone file (is provided)

        Returns
        --------
        gamma : complex numpy.ndarray
            complex  propagation constant
        z0 : numpy.ndarray
            complex port impedance
        '''
        def line2ComplexVector(s):
            return mf.scalar2Complex(numpy.array([k for k in s.strip().split(' ')
                                                if k != ''][self.rank*-2:],
                                                dtype='float'))

        with open(self.filename) as f:
            gamma, z0 = [],[]

            for line in f:
                if '! Gamma' in line:
                    gamma.append(line2ComplexVector(line.replace('! Gamma', '')))
                if '! Port Impedance' in line:
                    z0.append(line2ComplexVector(line.replace('! Port Impedance', '')))

            # If the file does not contain valid port impedance comments, set to default one
            if len(z0) == 0:
                z0 = self.resistance
                #raise ValueError('Touchstone does not contain valid gamma, port impedance comments')


        return numpy.array(gamma), numpy.array(z0)

def s2p_eff(file_path, input_freq, use_plot=False, r_range=None):
    touchstoneFile = Touchstone(file_path)
    freq_unit = touchstoneFile.frequency_unit
    freq_list, s_array = touchstoneFile.get_sparameter_arrays()
    index = find_nearest_index(input_freq, freq_list)
    use_freq = freq_list[index]
    s_param = s_array[index]

    S11 = s_param[0][0]
    S21 = s_param[1][0]
    S12 = s_param[0][1]
    S22 = s_param[1][1]


    # S(50) -> F
    A = (1 + S11) * (1 - S22) + S12 * S21
    B = (1 + S11) * (1 + S22) - S12 * S21
    C = (1 - S11) * (1 - S22) - S12 * S21
    D = (1 - S11) * (1 + S22) + S12 * S21

    Z = float(touchstoneFile.resistance)
    A = A / (2.0 * S21)
    B = B * Z / (2.0 * S21)
    C = C / (2.0 * S21 * Z)
    D = D / (2.0 * S21)

    eff_inv = (math.sqrt(4 * (A * C.conjugate()).real * (B * D.conjugate()).real -
                        ((A * D.conjugate() - B * C.conjugate()).imag)**2) + (A * D.conjugate() + B * C.conjugate()).real)
    eff = 1 / (math.sqrt(4 * (A * C.conjugate()).real * (B * D.conjugate()).real -
                        ((A * D.conjugate() - B * C.conjugate()).imag)**2) + (A * D.conjugate() + B * C.conjugate()).real)

    q = (A * D.conjugate() - B * C.conjugate()).imag / (2 * (A * C.conjugate()).real)
    F = (A * C.conjugate()).real * q * q - (A * D.conjugate() - B * C.conjugate()).imag * q + (B * D.conjugate()).real

    load_r = math.sqrt(F / ((A * C.conjugate()).real))
    load_i = (A * D.conjugate() - B * C.conjugate()).imag / (2 * (A * C.conjugate()).real)
    opt_load = load_r + load_i * 1.j
    input_impedance =  (A * opt_load + B) / (C * opt_load + D)

    print('{} {}'.format(use_freq, freq_unit))
    print('s_parameter: \n {} \n'.format(s_param))
    print('A * C.conj = {}'.format((A * C.conjugate()).real))
    print('efficiency: {} [%]'.format(eff*1e2))
    print('opt_load: {} + {}j [ohm]'.format(load_r, load_i))
    print('input impedance: {}'.format(input_impedance))

    if use_plot:
        if r_range is None:
            r_range = [0, load_r * 10.]
        div = 100
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(2.5, 2.5), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        load_list = numpy.linspace(r_range[0], r_range[1], div)
        eff_list = load_list / ((A * C.conjugate()).real * load_list**2 + (A * D.conjugate() + B * C.conjugate()).real * load_list + (B * D.conjugate()).real)
        ax.plot(load_list, eff_list*1e2, '-')
        ax.set_xlabel(r'Load')
        ax.set_ylabel(r'Efficiency')
        ax.set_ylim(0, 100)
        fig.tight_layout()
        plt.show()


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='s2p file to efficiency')
    parser.add_argument('file_path', help='s2p file path')
    parser.add_argument('-f', '--freq', default='6.78M', help='Frequency')
    parser.add_argument('-p', '--plot', action='store_true', help='Set flag if plot efficiency graph')
    parser.add_argument('--rrange', nargs='*', help='Load range in efficiency graph')
    args = parser.parse_args()

    file_path = args.file_path

    input_freq = parse_number_with_unit(args.freq)
    if input_freq is None:
        raise Exception('2th arg is frequency. ex: 6780000 or 6.78M')

    r_range=args.rrange
    if r_range:
        if len(r_range) != 2:
            raise Exception('--rrange arg is invalid. ex: --rrange 10.2 20.5')
        r_range[0] = float(r_range[0])
        r_range[1] = float(r_range[1])

    s2p_eff(file_path, input_freq, use_plot=args.plot, r_range=r_range)


if __name__ == "__main__":
    main()
