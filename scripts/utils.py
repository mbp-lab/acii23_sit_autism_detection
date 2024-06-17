def extract_id_clip(filename: str) -> (str, str):
    """
    Extracts the participant ID and video clip number from a given filename.

    Parameters:
    - filename (str): The filename from which to extract information.

    Returns:
    - (str, str): A tuple containing the participant ID and video clip number extracted from the filename.

    Example:
    >> filename = "participant123_part_2.mp4"
    >> extract_id_clip(filename)
    ('participant123', '2')
    """
    participant_id = slice(filename.find('_'))
    video_clip = slice(filename.rfind('_') + 1, None)
    return filename[participant_id], filename[video_clip]
