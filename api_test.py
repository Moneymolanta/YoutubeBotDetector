import csv
import json
import os
import googleapiclient.discovery
import googleapiclient.errors

from dotenv import load_dotenv


# Load environment variables. (API keys).
load_dotenv()

YOUTUBE_API_KEY = os.getenv("API_KEY")
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]


def replace_non_ascii(s):
    return ''.join([i if ord(i) < 128 else '?' for i in s])

def main():

    youtube = googleapiclient.discovery.build(API_SERVICE_NAME, API_VERSION, developerKey=YOUTUBE_API_KEY) #use you

    comments_request = youtube.commentThreads().list(
        part="snippet, replies",
        videoId ="2X-AF7fOzW0",         # gangdam style
        textFormat="plainText",         # You can also use 'html' if you prefer
        maxResults=1000,               # Adjust as needed; note that responses are paginated
        order="relevance"  # This will fetch the top (most relevant) comments
    )

    video_request = youtube.videos().list(part = "snippet", id = "2X-AF7fOzW0", maxResults = 1)

    video_response = video_request.execute()
    comments_response = comments_request.execute()
    next_page = youtube.commentThreads().list_next(comments_request, comments_response).execute()

    video_items = video_response['items']
    video_title = video_items[0]['snippet']['title']
    video_title = replace_non_ascii(video_title)
    items = comments_response["items"]
    items = items + next_page["items"]

    print(type(comments_response))

    print(len(comments_response["items"]))
    with open("validation_data.csv", "w", encoding="utf-8") as file:
        file.write("COMMENT_ID,AUTHOR,DATE,CONTENT,VIDEO_NAME,CLASS\n")
        for comment in items:
            comment_id = comment["snippet"]["topLevelComment"]["id"]
            author = comment["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
            date = comment["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
            content = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

            author = author.replace("@", "")
            content = content.replace(",", "")
            content = content.replace("\n", " ")
            date = date.replace("Z", "")
            line = f'{comment_id},{author},{date},{content},{video_title},2\n'
            file.write(line)


def is_within_one_year(date:str, reference_date:str) -> bool:
    """ Checks if date is within one year of reference date. """
    target_year = int(date[:4])
    reference_year = int(reference_date[:4])
    if abs(target_year - reference_year) > 5:
        return False
    return True

def pull_dataset(youtube, video_id: str, output_name: str, comment_limit: int):
    """
    Creates a dataset from a youtube video where comments are limited to within a year of the video being posted.
    Args:
        youtube: YouTube API client
        video_id: video id
        output_name: output file name
        comment_limit: maximum number of comments to retrieve
    """

    # Get video info. Contains post date and video title.
    video_info_request = youtube.videos().list(part="snippet", id=video_id, maxResults=1)
    video_info_response = video_info_request.execute()

    video_items = video_info_response['items']
    video_title = video_items[0]['snippet']['title']
    video_title = replace_non_ascii(video_title)
    video_date = video_items[0]['snippet']['publishedAt'].replace("Z", "")

    # Get comments for the video. Returns a list of comments.
    comments_request = youtube.commentThreads().list(
        part="snippet, replies",
        videoId=video_id,
        textFormat="plainText",
        maxResults=1000,
        order="relevance"
    )

    # Loop until there is a max number of comments grabbed.
    loops = 0
    count: int = 0
    lines = []
    print(f'Retrieving comments for video {video_title}...')
    while count < comment_limit and loops < 800:
        print(f'\tLoop {loops}. Count: {count}...')
        # Call API.
        comments_response = comments_request.execute()

        # Check if the request was successful.
        if comments_response is None:
            break

        # Extract list of items from request.
        items = comments_response["items"]

        for comment in items:
            # Check limit.
            if count >= comment_limit:
                break

            # Parse comment.
            date = comment["snippet"]["topLevelComment"]["snippet"]["publishedAt"].replace("Z", "")
            if is_within_one_year(date, video_date):
                # Extract info
                comment_id = comment["snippet"]["topLevelComment"]["id"]
                author = comment["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
                content = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

                # Clean values
                author = author.replace("@", "")
                content = content.replace(",", "")
                content = content.replace("\n", " ")
                content = content.replace("\t", " ")
                content = content.replace("\r", " ")
                date = date.replace("Z", "")

                # Add to list
                line = f'{comment_id},{author},{date},{content},{video_title},2\n'
                lines.append(line)
                count += 1
        loops += 1

        # Create new request for next loop.
        comments_request = youtube.commentThreads().list_next(comments_request, comments_response)

    # Write output.
    if not output_name.strip().endswith(".csv"):
        output_name += ".csv"
    with open(output_name, "w", encoding="utf-8") as file:
        file.write("COMMENT_ID,AUTHOR,DATE,CONTENT,VIDEO_NAME,CLASS\n")
        for line in lines:
            file.write(line)


if __name__ == "__main__":
    # main()

    # List taken from MTV Music Video award of the year list.
    # https://en.wikipedia.org/wiki/MTV_Video_Music_Award_for_Video_of_the_Year
    videos = [
        ('CvBfHwUxHIk', '2007-Rihanna_Umbrella'),
        ('u4FF6MpcsRw', '2008-Britney_Spears_Piece_of_Me'),
        ('4m1EFMoRFvY', '2009-Beyonce_Single_Ladies'),
        ('qrO4YZeyl0I', '2010-Lady_Gaga_Bad_Romance'),
        ('QGJuMBdaqIw', '2011-Katy_Perry_Firework'),
        ('tg00YEETFzg', '2012-Rihanna_We_Found_Love'),
        ('uuZE_IRwLNI', '2013-Justin_Timberlake_Mirror'),
        ('My2FRPA3Gf8', '2014-Miley_Cyrus_Wrecking_Ball'),
        ('QcIy9NiNbmo', '2015-Taylor_Swift_Bad_Blood'),
        ('WDZJPJV__bQ', '2016-Beyonce_Formation'),
        ('tvTRZJ-4EyI', '2017-Kendrick_Lamar_Humble'),
        ('HCjNJDNzw8Y', '2018-Camilla_Cabello_Havana'),
        ('Dkk9gvTmCXY', '2019-Taylor_Swift_You_Need_To_Calm_Down'),
        ('4NRXx6U8ABQ', '2020-The_Weeknd_Blinding_Lights'),
        ('6swmTBVI83k', '2021-Lil_Nas_X_Montero'),
        ('tollGa3S0o8', '2022-Taylor_Swift_All_Too_Well_Short_Film'),
        ('b1kbLwvqugk', '2023-Taylor_Swift_Anti_Hero'),
        ('q3zqJs7JUCQ', '2024-Taylor_Swift_Fortnight'),
    ]

    # Init Google API client.
    youtube = googleapiclient.discovery.build(API_SERVICE_NAME, API_VERSION, developerKey=YOUTUBE_API_KEY) #use you

    # Fetch results for each page.
    for video in videos:
        pull_dataset(youtube, video[0], video[1], 200)
