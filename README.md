# pomelo-disease-classification
For BSCS Thesis by Agawin, Apostol, and Cabangbang

# CSV Tracker Statuses
Possible Values:
    1. Unprocessed  - No action(s) have been done on image.
    2. Extracted    - Initial extraction of pomelo. Skipped by extractor.
    3. Incorrect    - May need mask or center point override. Repeated by extractor.
    4. Partial      - Has elements that need to be manually removed. Skipped by extractor.
    5. Processed    - Ready for machine learning model. Skipped by extractor.
    6. Unusable     - Is either a worse duplicate of another image or has more than 1 disease.  Skipped by extractor.
