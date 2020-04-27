import os
from encodings.aliases import aliases

import cv2


def draw_box_on_image(image_data, box, color=(0, 0, 255), info=False):
    left, top, width, height = box
    right = left + width
    bottom = top + height
    point_left_top = (int(left), int(top))
    point_right_bottom = (int(right), int(bottom))
    processed_image_data = image_data.copy()
    cv2.rectangle(img=processed_image_data, pt1=point_left_top, pt2=point_right_bottom, color=color)
    if info:
        print(box, 'is drawn.')
    return processed_image_data


def check_encoding_supported(encoding):
    is_supported = False
    for key, value in enumerate(aliases.items()):
        if encoding in list(value):
            is_supported = True
            break
    return is_supported


def read_file(file_path, text_or_binary='text', encoding='utf_8'):
    if not os.path.exists(file_path):
        raise ValueError('"file_path": file does not exist.')
    if text_or_binary == 'text':
        read_mode = 'r'
    elif text_or_binary == 'binary':
        read_mode = 'rb'
    else:
        raise ValueError('"text_or_binary": parameter can be "text" or "binary".')
    if not check_encoding_supported(encoding):
        raise ValueError('"encoding": unsupported encoding.')
    lines = []
    with open(file=file_path, mode=read_mode, encoding=encoding) as file:
        for line_counter, line in enumerate(file):
            lines.append(line)
    return lines


def save_file(file_path, lines, text_or_binary='text', encoding='utf_8'):
    if not isinstance(lines, list):
        raise ValueError('"lines": parameter should be a list.')
    if text_or_binary == 'text':
        write_mode = 'w'
    elif text_or_binary == 'binary':
        write_mode = 'wb'
    else:
        raise ValueError('"text_or_binary": parameter can be "text" or "binary".')
    if not check_encoding_supported(encoding):
        raise ValueError('"encoding": unsupported encoding.')
    with open(file=file_path, mode=write_mode, encoding=encoding) as file:
        for line in lines:
            line += '\n'
            file.write(line)


def get_files_in_directory(directory):
    files = []
    filenames = os.listdir(directory)
    for filename in filenames:
        file = os.path.join(directory, filename)
        files.append(file)
    return files


def filter_files_by_extensions(files, file_extensions):
    if len(file_extensions) == 0:
        return files
    filtered_files = []
    for file in files:
        file_extension = file.split('\\')[-1].split('.')[-1]
        if file_extension in file_extensions:
            filtered_files.append(file)
    return filtered_files


def track(sequence_directory, tracker_names, instance_with_templates, image_extension='png',
          output_extension='csv', output_separator=',', info=False):
    # opencv trackers
    opencv_trackers = dict()
    opencv_trackers['boosting'] = cv2.TrackerBoosting_create
    opencv_trackers['mil'] = cv2.TrackerMIL_create
    opencv_trackers['kcf'] = cv2.TrackerKCF_create
    opencv_trackers['tld'] = cv2.TrackerTLD_create
    opencv_trackers['medianflow'] = cv2.TrackerMedianFlow_create
    opencv_trackers['mosse'] = cv2.TrackerMOSSE_create
    opencv_trackers['csrt'] = cv2.TrackerCSRT_create

    if not os.path.exists(sequence_directory):
        raise ValueError('Path does not exists:', sequence_directory)

    # trackers
    trackers = dict()
    for tracker_name in tracker_names:
        if str(tracker_name) not in opencv_trackers.keys():
            raise ValueError('Unsupported OpenCV tracker:', str(tracker_name))
        else:
            tracker_method = opencv_trackers[tracker_name]
            trackers[tracker_name] = tracker_method

    # paths
    paths = dict()
    parent_directory = os.path.dirname(sequence_directory)
    sequence_name = parent_directory.split('\\')[-1]
    paths['results'] = os.path.join(parent_directory, 'results')
    for tracker_name in trackers.keys():
        paths['results_' + str(sequence_name) + '_' + str(tracker_name)] = \
            os.path.join(paths['results'], str(sequence_name) + '_' + str(tracker_name))
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)

    # sequence image paths
    file_paths = get_files_in_directory(directory=sequence_directory)
    image_paths = filter_files_by_extensions(files=file_paths, file_extensions=[image_extension])
    number_of_images = len(image_paths)

    # sequence image path list
    frame_with_image_paths = list()
    for image_path in image_paths:
        image_filename = image_path.split('\\')[-1].split('.')[0]
        frame_name = int(image_filename.split('_')[-1])
        frame_with_image_path = frame_name, image_path
        frame_with_image_paths.append(frame_with_image_path)

    # results
    trackers_with_instances = dict()
    for tracker_name in trackers.keys():
        tracker_method = trackers[tracker_name]
        trackers_with_instances[tracker_name] = dict()
        for instance_name in instance_with_templates.keys():
            tracks = list()
            tracker = tracker_method()
            trackers_with_instances[tracker_name][instance_name] = tracker, tracks

    # info
    initialize_info = 'sequence={sequence} tracker={tracker} instance={instance}'
    track_info = 'sequence={sequence} tracker={tracker} instance={instance} frame={frame} box={box}'

    # tracking templates
    for index in range(number_of_images):
        frame_name, image_path = frame_with_image_paths[index]
        image_data = cv2.imread(filename=image_path, flags=cv2.IMREAD_COLOR)
        if info:
            print(image_path, 'is loaded.')
        for tracker_name in trackers_with_instances.keys():
            for instance_name in trackers_with_instances[tracker_name].keys():
                tracker, tracks = trackers_with_instances[tracker_name][instance_name]
                if index == 0:
                    box = instance_with_templates[instance_name]
                    success = tracker.init(image_data, box)
                    if info:
                        success_info_text = 'LOG: initialize' if success else 'ERROR: initialize'
                        initialize_info_text = initialize_info.format(sequence=sequence_name,
                                                                      tracker=tracker_name,
                                                                      instance=instance_name)
                        print(success_info_text, initialize_info_text)

                else:
                    success, box = tracker.update(image_data)
                    if info:
                        success_info_text = 'LOG: track' if success else 'ERROR: track'
                        track_info_text = track_info.format(sequence=sequence_name,
                                                            tracker=tracker_name,
                                                            instance=instance_name,
                                                            frame=frame_name,
                                                            box=box)
                        print(success_info_text, track_info_text)
                frame_with_box = frame_name, box
                tracks.append(frame_with_box)
                trackers_with_instances[tracker_name][instance_name] = tracker, tracks

    # generating output files
    for tracker_name in trackers_with_instances.keys():
        for instance_name in trackers_with_instances[tracker_name].keys():
            _, tracks = trackers_with_instances[tracker_name][instance_name]
            lines = list()
            for instance_track in tracks:
                frame_name, box = instance_track
                left, top, width, height = box
                line = '{frame}{separator}{left}{separator}{top}{separator}{width}{separator}{height}' \
                    .format(frame=frame_name, left=left, top=top, width=width,
                            height=height, separator=output_separator)
                lines.append(line)
            file_name = str(sequence_name) + "_" + str(tracker_name) + '_' + str(instance_name) + '.' + output_extension
            file_path = os.path.join(paths['results_' + str(sequence_name) + '_' + str(tracker_name)], file_name)
            save_file(file_path=file_path, lines=lines, text_or_binary='text', encoding='utf_8')
            if info:
                print(file_path, 'is generated.')


# main
def main():
    # parameters
    info = True
    image_extension = 'png'
    output_extension = 'csv'
    output_separator = ','

    # sequences
    sequences = dict()
    sequences['sequence_1'] = 'C:\\dataset\\sequence_1'
    sequences['sequence_2'] = 'C:\\dataset\\sequence_2'

    # trackers
    tracker_names = ['boosting', 'mil', 'kcf', 'tld', 'medianflow', 'mosse', 'csrt']

    # initial interpolated bounding boxes
    instance_with_templates = dict()
    instance_with_templates['1'] = 586.0, 770.0, 25.0, 22.0
    instance_with_templates['2'] = 452.0, 585.0, 15.0, 25.0

    # processing sequences
    for sequence_name in sequences.keys():
        sequence = sequences[sequence_name]
        track(sequence_directory=sequence, tracker_names=tracker_names,
              instance_with_templates=instance_with_templates, image_extension=image_extension,
              output_extension=output_extension, output_separator=output_separator, info=info)


if __name__ == '__main__':
    main()
