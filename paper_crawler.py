import argparse
import os, json
import pandas as pd
import requests
from tqdm import tqdm

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--src_path', type=str, default='./paperurls_withurltype.csv', help='url dataframe path')
	parser.add_argument('--conference_id', type=str, default='', help='conference_id')
	parser.add_argument('--output_path', type=str, default='./paper', help='json output path')
	parser.add_argument('--start_index', type=int, default=0, help='start index')

	args = parser.parse_args()
	src_path = args.src_path
	conference_id = args.conference_id
	output_path = args.output_path
	start_index = args.start_index

	if len(conference_id)!=10:
		raise AssertionError("invalid conference id")

	if not os.path.isdir(output_path):
		os.mkdir(output_path)
	
	data = pd.read_csv(src_path)
	data = data[data['ConferenceSeriesId']==int(conference_id)]

	paper_title = list(data['PaperTitle'].unique())

	url_format = 'https://api.openalex.org/works?filter=title.search:'

	for idx, title in enumerate(tqdm(paper_title[start_index:])):

		idx += start_index

		title_replaced = title.replace(' ','+')
    
		target_url = url_format + title_replaced

		r = requests.get(target_url)

		file_name = os.path.join(output_path,'{}.json'.format(idx))

		with open(file_name, 'w') as f:
		    json.dump(r.json(), f)

if __name__=='__main__':
	main()
