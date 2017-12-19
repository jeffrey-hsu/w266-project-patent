'''For class demonstration during our presentation.'''

from similarity_query import similarity_query, show_best

patent_no = 3930278
max_text_length = 2000

best = similarity_query(patent_no)
print('\n\nThe first match is the patent itself: %i' % patent_no)
print('Text:')
print(best[0][1])

print('\n\n')
show_best(best)

raw_input('\nPlease press enter to continue.\n')

for patent in best[1:4]:
    print(patent[1][:max_text_length])
    raw_input('\nPlease press enter to continue.\n')
