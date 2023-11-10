import csv

def parse_sign_annotations_export_to_csv(file_name):
    content = []

    with open(file_name, 'r') as file:
        image_nr = 0

        for line in file:
            name_end = line.find(':')
            if name_end == -1:
                print('ERROR: file incorrect!')
                break

            im_name = line[:name_end]
            image_nr += 1

            image_data = {
                'image_name': im_name,
                'sign_type': [],
                'sign_status': [],
                'bbox': [],
                'sign_c': [],
                'sign_size': [],
                'aspect_ratio': [],
            }

            if len(line) == name_end:
                content.append(image_data)
            else:
                nr_signs = 0
                sign_end = line.find(';')

                while sign_end > 0:
                    if line[name_end + 1:sign_end] == 'MISC_SIGNS':
                        nr_signs += 1
                        line = line[sign_end + 1:]
                        sign_end = line.find(';')

                        sign_info = {
                            'signTypes': 'MISC_SIGNS',
                            'signStatus': 'N/A',
                            'signBB': '[-1, -1, -1, -1]',
                            'signC': '[-1, -1]',
                            'signSize': 0,
                            'aspectRatio': 0,
                        }

                        image_data['sign_type'].append(sign_info['signTypes'])
                        image_data['sign_status'].append(sign_info['signStatus'])
                        image_data['bbox'].append(sign_info['signBB'])
                        image_data['sign_c'].append(sign_info['signC'])
                        image_data['sign_size'].append(sign_info['signSize'])
                        image_data['aspect_ratio'].append(sign_info['aspectRatio'])

                    else:
                        nr_signs += 1
                        commas = [i for i, char in enumerate(line) if char == ',']
                        visibility = line[:commas[0]].split(':')[-1]

                        lrx = float(line[commas[0] + 2:commas[1]])
                        lry = float(line[commas[1] + 2:commas[2]])
                        ulx = float(line[commas[2] + 2:commas[3]])
                        uly = float(line[commas[3] + 2:commas[4]])
                        sign_type = line[commas[4] + 2:commas[5]]
                        sign_name = line[commas[5] + 2:sign_end]

                        line = line[sign_end + 1:]
                        sign_end = line.find(';')

                        sign_info = {
                            'signTypes': sign_name,
                            'signStatus': visibility,
                            'signBB': [ulx, uly, lrx, lry],
                            'signC': [(ulx + lrx) / 2, (uly + lry) / 2],
                            'signSize': min((ulx - lrx), (uly - lry)) ** 2,
                            'aspectRatio': max((ulx - lrx) / ((uly - lry) + 1e-6),
                                               (uly - lry) / ((ulx - lrx) + 1e-6)),
                        }

                        image_data['sign_type'].append(sign_info['signTypes'])
                        image_data['sign_status'].append(sign_info['signStatus'])
                        image_data['bbox'].append(str(sign_info['signBB']))
                        image_data['sign_c'].append(str(sign_info['signC']))
                        image_data['sign_size'].append(sign_info['signSize'])
                        image_data['aspect_ratio'].append(sign_info['aspectRatio'])

                content.append(image_data)
                
    output_file_name = file_name.split('.')[0]       
    csv_file = export_to_csv(content, output_file = f'{output_file_name}.csv')
    
    return csv_file
    
def export_to_csv(data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['image_name', 'sign_type', 'sign_status', 'bbox', 'sign_c', 'sign_size', 'aspect_ratio']
        writer.writerow(headers)

        for entry in data:
            for i in range(len(entry['sign_type'])):
                row = [
                    entry['image_name'],
                    entry['sign_type'][i],
                    entry['sign_status'][i],
                    entry['bbox'][i],
                    entry['sign_c'][i],
                    entry['sign_size'][i],
                    entry['aspect_ratio'][i],
                ]
                writer.writerow(row)
                
    return output_file