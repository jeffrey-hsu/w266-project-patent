'''Simply counts the number of patents in the
csv file.'''

from datetime import datetime as dt

def clump(filename):
    '''Sorts through the lines, combining them according
    to patent number, and outputs the joined text.'''

    # Initialize values
    last_patent_number = 0
    clump_text = ''

    with open(filename, 'r') as f:
    # line = [patent number, claim number, claim text, dependencies,
    # ind_flg, appl_id.]

        for i, line in enumerate(f):
            # Ignore the header
            if i == 0:
                pass

            else:
                # Retrieve patent number and text, according to format
                if '"' in line:
                    patent_no = line.split('"')[0].split(',')[0]
                    claim_text = line.split('"')[1]
                else:
                    patent_no = line.split(',')[0]
                    claim_text = line.split(',')[2]

                # Add to the string if it's the same patent as the last line
                if patent_no == last_patent_number:
                    clump_text = ' '.join((clump_text, claim_text))

                # Output the old line if a new patent is encountered,
                # and reset the values for patent number and text
                else:
                    if last_patent_number != 0:
                        yield last_patent_number, clump_text
                    last_patent_number = patent_no
                    clump_text = claim_text
    yield last_patent_number, clump_text # Output the last clump as well

base_file_path = '/home/cameronbell/'
patent_claims_file = ''.join((base_file_path, 'patent_data/patent_claims_fulltext.csv'))

print(dt.now())
print('Starting to iterate')
for i, clump in enumerate(clump(patent_claims_file)):
    if i%1000000 == 0:
        print('%i patents processed.' % i) 
    count = i

print('Finished. Total clumps: %i' % count)
print(dt.now())
