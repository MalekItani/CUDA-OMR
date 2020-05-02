import cv2
import numpy as np
import time
import glob
from mingus.midi import fluidsynth
import matplotlib.pyplot as plt
import imutils
import os

BPM = 120

DURATION_QUARTER_NOTE = 60/BPM

DURATION_EIGHTH_NOTE = DURATION_QUARTER_NOTE/2
DURATION_SIXTEENTH_NOTE = DURATION_EIGHTH_NOTE/2

DURATION_HALF_NOTE = 2 * DURATION_QUARTER_NOTE
DURATION_WHOLE_NOTE = 2 * DURATION_HALF_NOTE

def full_step(n):
    note = (n - 1) % 12
    if note == 2 or note == 7:
        return n + 1
    else:
        return n + 2

def advance(n, n_step):
    for i in range(n_step):
        n = full_step(n)
    return n

# Ignore this.
def classify(note, staff, clef='treble'):
    """
    note: Center of the note
    staff: array of 5 points denoting the position of the lines of the staff
    """
    staff_width = np.mean([staff[x] - staff[x-1] for x in range(1, len(staff))])
    note_increment = staff_width/2
    n = 44
    delta_n = int(round((staff[-1] - note[1])/note_increment))
    return advance(n, delta_n)

def classify2(note, staff, staff_thickness, staff_spacing, clef='treble'):
    """
    note: Center of the note
    staff: array of 5 points denoting the position of the lines of the staff
    """
    note_increment = (staff_thickness + staff_spacing)//2
    n = 44
    delta_n = int(round(( (staff[-1] + staff_thickness//2) - note[1])/note_increment))
    return advance(n, delta_n)

def play(note, duration=1):
    fluidsynth.play_Note(note)
    time.sleep(duration)


# Start here.
def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # img = cv2.blur(img, (1, 1))
    img_shape = 400
    # img = cv2.resize(img, dsize=(img_shape, int(img_shape/img.shape[1] * img.shape[0])))
    img = imutils.resize(img, height=img_shape)
    return img

def load_dictionary():
    dictionary = {}
    for path in glob.glob('templates/*.png'):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
        bin_img = (1 * (gray == 255)).astype(np.uint8)
        name = os.path.basename(path)[:-4]
        dictionary[name] = bin_img
    return dictionary

# Ignore this
def test():
    fluidsynth.init('sfs/soundfont.sf2', 'alsa')
    img = read_image('src/ode_to_joy.png')
    result = img.copy()

    t1 = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    lines = cv2.HoughLinesP(gray, 1, np.pi/2, 100, maxLineGap=15, )

    staff_tmp = []
    # print(lines)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) <= 1:
                staff_tmp.append(y2)
                cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 1)

    staff_tmp.sort()
    staff = []
    for i in range(len(staff_tmp)):
        if len(staff) == 0 or staff_tmp[i] - staff[-1] >= img.shape[1]/100:
            staff.append(staff_tmp[i])
    print(staff)
    notes = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1)
    if notes is not None:
        for (x, y, r) in notes:
            cv2.circle(result, (x, y), r, (0,0,255),1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 4))
    gray = cv2.erode(gray, kernel=kernel, iterations=1)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img, contours, -1, (0,255,0), 3)

    note_sequence = []
    if contours is not None:
        for contour in contours:
            # (cx, cy), radius = cv2.minEnclosingCircle(contour)
            # print(contour)
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(img, (cx, cy), 2, (0,0,255))
            note_sequence.append((cx, classify((cx, cy), staff)))
    note_sequence.sort()
    # note_sequence[2] = (note_sequence[2][0], 48)
    print(note_sequence)

    t2 = time.time()
    print(t2 - t1)

    cv2.imshow("gray", gray)
    
    cv2.imshow("Original", img)
    cv2.imshow("Result Image", result)
    cv2.waitKey(0)

    
    for note in note_sequence:
        play(note[1], 0.5)

    
    cv2.destroyAllWindows()

def sliding_window_argmax(arr, k):
    """
    Given a 1D array arr, computes the index of maximum value of this array.
    """
    # O(kn)
    max_sum = 0
    max_idx = (-1,-1)
    for i in range(k, len(arr)-k):
        acc = 0
        for j in range(-k, k+1):
            acc += arr[i+j]
        if acc > max_sum:
            max_sum = acc
            max_idx = (i-k, i+k)
    return max_idx, max_sum

def compute_staff(gray):
    """
    Extract staff parameters from input image.
    Staff parameters are staff width and staff spacing.
    This is done by creating a histogram of consecutive black an white pixels, and 
    taking the lengths of black pixels which occur most as staff thickness, and lengths
    of white pixels which occur most as staff spacing.
    This algorithm is described in Optical Music Recognition using Projections.
    """
    # Initialize histograms
    white_hist = np.zeros(gray.shape[0]+1)
    black_hist = np.zeros(gray.shape[0]+1)
    # Loop over columns
    for j in range(gray.shape[1]):
        i = 0
        while i < gray.shape[0]:
            # Compute length of consecutive sequence of white pixels
            sequence_length = 0
            while i < gray.shape[0] and gray[i,j] > 0:
                sequence_length += 1
                i += 1
            if sequence_length > 0:
                white_hist[sequence_length] += 1

            # Compute length of consecutive sequence of black pixels
            sequence_length = 0
            while i < gray.shape[0] and gray[i,j] == 0:
                sequence_length += 1
                i += 1
            if sequence_length > 0:
                black_hist[sequence_length] += 1

    staff_thickness = sliding_window_argmax(black_hist, 1)[0]
    staff_spacing = sliding_window_argmax(white_hist, 1)[0]
    return staff_thickness[0], staff_spacing[1]

def find_staves(I, staff_thickness, staff_spacing):
    """
    Given a binary image I, locates the staves in the image.
    A staff is a list of 5 staff y-positions.
    """
    staff_positions = []
    img = I
    # Define the score of a row as the number of matching pixels along all columns of this row
    # Formally, maintain an array score, where score[i] = sum of scores over all columns of row i
    score = np.zeros(img.shape[0])
    
    # Loop over every pixel
    for j in range(img.shape[1]):
        for i in range(img.shape[0] - staff_thickness):
            score[i] += np.sum(1 - img[i:i+staff_thickness, j])
            assert(score[i] >= 0)
    
    # Take rows that are above a certain threshold and skip by one template
    # Threshold is 80%
    # Adaptive staff prediction: If a staff line is expected, then decrease confidence threshold.
    # This helps reduce staff lines which are left undetected, while also reducing false positives.
    confidence = 0.8
    threshold = img.shape[1] * staff_thickness
    row = 0
    staff = []
    while row < img.shape[0]:
        if score[row] > threshold * confidence:
            staff.append(row+staff_thickness//2)
            if len(staff) == 5:
                staff_positions.append(staff)
                staff = []
                confidence = 0.8
            row += staff_spacing - 2
            if confidence == 0.8:
                confidence = 0.6
        else:
            row += 1
    return staff_positions

def draw_staff(img, staff, staff_thickness, staff_spacing, color=(255, 0, 0), thickness=None):
    if thickness is None:
        thickness = staff_thickness
    for y in staff:
        cv2.line(img, (0, y), (img.shape[1], y), color, thickness)

def segment_by_staves(img, staves, staff_thickness, staff_spacing):
    """
    Splits tracks by the staff positions.
    Returns list of tracks, list of track y-offsets
    """
    track_bounds = []
    for staff in staves:
        # Consider two imaginary staff lines above and below to account for notes
        # above and below the staff. (May need to increase this)
        y1 = max(staff[0] - 2*(staff_thickness + staff_spacing), 0)
        y2 = min(staff[-1] + 2*(staff_thickness + staff_spacing), img.shape[0])
        track_bounds.append( (y1, y2) )
    return track_bounds

def get_projection(img, start=0, end=-1, axis='X'):
    """
    Returns the img projection (sum of black pixels) from start to end along a specified axis.
    """
    if end == -1:
        if axis == 'X':
            end = img.shape[1]
        else:
            end = img.shape[0]
    result = np.zeros(end - start)

    if axis == 'X':
        for j in range(start, end):
            for i in range(img.shape[0]):
                if img[i][j] == 0:
                    result[j-start] += 1
    elif axis == 'Y':
        for i in range(start, end):
            for j in range(img.shape[1]):
                if img[i][j] == 0:
                    result[i-start] += 1
    else:
        raise Exception("Invalid value for parameter \'axis\'. It should be either \'X\' or \'Y\'.")
    return result

def get_interesting_intervals(proj, threshold):
    """
    Returns a list of intervals where proj is greater than some threshold.
    """
    boundaries = []
    i = 0
    while i < len(proj):
        if proj[i] < threshold or proj[i] >= np.max(proj) - 1:
            i += 1
        else:
            boundary = (i,)
            while i < len(proj) and proj[i] >= threshold:
                i += 1
            boundary += (i,)
            boundaries.append(boundary)
    return boundaries

def find_all_symbols(img, staff_thickness, staff_spacing, draw_projection_plots=False):
    """
    Returns bounding boxes over all symbols in the image.
    Note that the coordinates are relative so be sure to add the y-offset for multitrack images.
    """
    xproj = get_projection(img, axis='X')
    yproj = get_projection(img, axis='Y')

    if draw_projection_plots:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.set_title("X Projection")
        x = np.arange(0, len(xproj), 1)
        ax1.fill_between(x, xproj)
        ax2.set_title("Y Projection")
        x = np.arange(0, len(yproj), 1)
        ax2.fill_between(yproj, x)
        ax2.invert_yaxis()
        fig.tight_layout()

    vertical_boundaries = get_interesting_intervals(xproj, threshold=staff_thickness)
    
    objects = []

    for vboundary in vertical_boundaries:
        yproj = get_projection(img[:, vboundary[0]:vboundary[1]], axis='Y')
        horizontal_boundaries = get_interesting_intervals(yproj, threshold=staff_thickness//2)
        for hboundary in horizontal_boundaries:
            objects.append((vboundary[0], hboundary[0], vboundary[1], hboundary[1]))
    
    return objects

# Ignore this.
def recognize_isolated_note(img, symbol, staff_thickness, staff_spacing):
    """
    Given a picture of an isolated note, i.e. without beams, computes the y-position of its
    head as well as the duration encoded by this note.
    It does so using the following algorithm:
    If the symbol is more wide than it is high, it is a whole note.
    Otherwise, do:
    1 - Compute local x-projection
    2 - Find the position of maximum element, which should correspond to the position of the 
    note stem. Flagged notes (eighth, sixteenth) should have this near the center,
    while other stemmed notes (half, quarter) should have it near some edge.
    Moreover, given in what half this maximum lies, we can deduce the orientation of the note
    in order to extract the note head.
    3 - If it is flagged, assume it is eighth.
    4 - Otherwise, compute the number of black cells in the flag head using any projection
    with the appropriate start and end positions.
    """

    # Obtain the boundaries from the symbol
    x1, y1, x2, y2 = symbol
    
    # Store the height and width as dx, dy
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Check if they're too small, this could be due to a false detection in a previous step
    if dx < staff_spacing//2 or dy < staff_spacing:
        return []

    # Take a slice of the image by the bounding box of the symbol
    symbol_img = img[y1:y2, x1:x2]

    # Compute the x-projection and the position of the maximum element.
    x_proj = get_projection(symbol_img)
    x_proj_argmax = sliding_window_argmax(x_proj, 1)[0][0]

    # Find the center of the note head. 
    # If the maximum computed earlier lies in the first half, then the note stem points downwards,
    # so adjust the y-center accordingly.  
    cx = (x1 + x2)//2
    cy = y2 - (staff_spacing + staff_thickness)//2 - 1
    if x_proj_argmax < len(x_proj)//2:
        cy = y1 + (staff_spacing + staff_thickness)//2 + 1

    # Check if it is more wide than it is high
    if dx > dy:
        duration = DURATION_WHOLE_NOTE
    else:
        # Check if the position of the max lies near the center for the note to be flagged.
        if x_proj_argmax >= len(x_proj)//3 and x_proj_argmax <= 2*len(x_proj)//3:
            duration = DURATION_EIGHTH_NOTE
        else:
            x_offset = staff_thickness
            y_offset = staff_thickness
            
            symbol_notehead_size = staff_spacing

            roi = [[x_offset, symbol_img.shape[1] - x_offset], [symbol_img.shape[0] - y_offset - symbol_notehead_size, symbol_img.shape[0]-y_offset]]

            if x_proj_argmax < len(x_proj)//2:
                roi[1] = [y_offset, y_offset + symbol_notehead_size]

            y_proj = get_projection(symbol_img[:,roi[0][0]:roi[0][1]], start = roi[1][0], end=roi[1][1], axis='Y')
            num_blacks = np.sum(y_proj)
            
            if num_blacks >= (dx - 2 * x_offset) * (symbol_notehead_size - y_offset) * 0.9:
                duration = DURATION_QUARTER_NOTE
            else:
                duration = DURATION_HALF_NOTE

            # print((roi[0][0], roi[1][1]))
            # print((roi[0][1], roi[1][0]))

            # cv2.rectangle(symbol_img, (roi[0][0], roi[1][0]), (roi[0][1], roi[1][1]), (255,255,255), -1)
            # cv2.rectangle(symbol_img, (0, symbol_img.shape[0]), (symbol_img.shape[1], symbol_img.shape[0] - staff_spacing), (255,255,255), -1)

    # cv2.imshow('', symbol_img)
    return [((cx, cy), duration)]

# Ignore this.
def recognize_note_symbol(img, symbol, offset, staff_thickness, staff_spacing):
    x1, y1, x2, y2 = symbol
    y1 += offset
    y2 += offset
    dx = x2 - x1
    if dx > 3*staff_spacing:
        return []
    else:
        return recognize_isolated_note(img, (x1, y1, x2, y2), staff_thickness, staff_spacing)

# TODO: Implement
def _match(I, symbol, mask):
    """
    Returns the accuracy score and position of a single symbol matched with a template mask.
    """
    x1, y1, x2, y2 = symbol
    mask_height, mask_width = mask.shape
    nrows, ncols = I.shape
    score = 0
    pos = (-1, -1)
    
    max_ratio = 1.4
    if (x2 - x1) / mask_width <= max_ratio and (y2 - y1) / mask_height <= max_ratio:
        for i in range(y1, max(y2 - mask_height, y1 + 1) ):
            for j in range(x1, max(x2 - mask_width, x1 + 1)):
                tmp = 0
                for x in range(mask_height):
                    for y in range(mask_width): 
                        if i + x < nrows and j + y < ncols:
                            tmp += (I[i+x, j+y] == mask[x,y])
                if tmp > score:
                    score = tmp
                    pos = (i, j)
    return score/(mask_height * mask_width), pos

def match_symbol(I, symbol, dictionary):
    scores = []
    for name, mask in dictionary.items():
        score, pos = _match(I, symbol, mask)
        scores.append( (score, name, pos) )
    scores.sort(reverse=True)
    print(scores)
    if scores[0][0] < 0.5:
        return '-1', -1
    return scores[0][1], scores[0][2]

def compute_runs(I, axis='X'):
    """
    Given an input binary image I, computes an output image res, where
    res[i, j] = longest run ending at this pixel.
    A run is defined as a consecutive sequence of black pixels.
    If I[i, j] = 1, i.e. represents a white pixel, then res[i, j] = 0
    """ 
    nrows, ncols = I.shape
    res = np.zeros_like(I)
    if axis == 'X':
        for i in range(nrows):
            current_sequence = 0
            for j in range(ncols):
                if I[i, j] == 0: # Black pixel
                    current_sequence += 1
                else:
                    current_sequence = 0
                res[i, j] = current_sequence
    elif axis == 'Y':
        for j in range(ncols):
            current_sequence = 0
            for i in range(nrows):
                if I[i, j] == 0: # Black pixel
                    current_sequence += 1
                else:
                    current_sequence = 0
                res[i, j] = current_sequence
    return res

def remove_staff(I, staff, staff_thickness):
    # Algorithm discussed in Robust and ...
    nrows, ncols = I.shape
    res = I.copy()

    # Compute Iv
    Iv = compute_runs(I, axis='Y')
    
    # For every staff y-position, go over all columns and remove the run if 
    # its length is <= staff_thickness + 2
    for x in staff:
        for j in range(ncols):
            
            if Iv[x, j] ==  0:
                continue

            x2 = x
            while x2 < nrows and Iv[x2, j] > 0:
                x2 += 1
            
            if Iv[x2-1, j] <= staff_thickness + 2:
                x1 = x2 - Iv[x2-1, j]
                while x1 < x2:
                    res[x1, j] = 1
                    x1 += 1
    return res

def find_vertical_lines(I, staff_thickness, staff_spacing):
    # Algorithm discussed in Robust and ...
    nrows, ncols = I.shape
    
    # Compute Iv
    Iv = compute_runs(I, axis='Y')
    
    # Compute Ih (Used to account for skewness. Implement iza elak khele2.)
    Ih = compute_runs(I, axis='X')
    
    # The paper assumed this to be at most 5, this didn't work so I'm making it adaptive
    expected_segment_width = 3 * staff_thickness//2
    if expected_segment_width % 2 == 0:
        expected_segment_width -= 1
    Nl = np.zeros(expected_segment_width + 4)
    Nl[:2] += 1/4
    Nl[-2:] += 1/4
    mask_radius = len(Nl)//2

    # Compute Il(x, y) = I(x, y) * sum_{-4}^{4}(I(x, y+j) * N(j))
    Il = np.zeros_like(I)
    for i in range(nrows):
        for j in range(ncols):
            if I[i, j] == 0:
                pixval = 0
                
                # TODO: Optimize?
                for k in range(-mask_radius, mask_radius+1):
                    if j + k >= 0 and j + k < ncols:
                        pixval += I[i, j + k] * Nl[mask_radius + k]
                
                Il[i, j] = pixval

    potential_vertical_lines = []

    # Find the largest run in every column, and check if it validates conditions (2) and (3)
    for j in range(ncols):
        largest_run = 0
        xh, xb = (0, 0) # Extremities
        for i in range(nrows):
            if Iv[i, j] > largest_run:
                largest_run = Iv[i, j]
                xh, xb = (i-largest_run, i)
            
        if largest_run > 2 * staff_spacing: # Ignore this for now? and np.sum(Il[xh, xb])/largest_run > 1/4:
            potential_vertical_lines.append((j, xh, xb))

    # Filter out lines that are within 2/5 staff spacing of one another so that each line returns
    # only vertical segment
    vertical_lines = []
    for line in potential_vertical_lines:
        if len(vertical_lines) == 0 or line[0] - vertical_lines[-1][0] > 2/5 * staff_spacing:
            vertical_lines.append(line)

    # cv2.imshow("IMM", (Iv//2).astype(np.uint8))

    return vertical_lines


# Ignore this.
def algorithm1(img, gray, staves, tracks, offsets, staff_thickness, staff_spacing, draw_projection_plots):
    note_sequence = []
    t1 = time.time()
    for staff, track, y_offset in zip(staves, tracks, offsets):
        symbols = find_all_symbols(track, staff_thickness, staff_spacing, draw_projection_plots=draw_projection_plots)
        for x1, y1, x2, y2 in symbols:
            cv2.rectangle(img, (x1, y1+y_offset), (x2, y2+y_offset), (0,255,0), 2)
        for symbol in symbols:
            notes = recognize_note_symbol(gray, symbol, y_offset, staff_thickness, staff_spacing)
            for ((cx, cy), duration) in notes:
                cv2.circle(img, (cx, cy), 1, (0,0,255))
                cv2.putText(img, str(duration), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                note = classify2((cx, cy), staff, staff_thickness, staff_spacing)
                note_sequence.append((note, duration))

    t2 = time.time()
    print("Time taken to find and classify all symbols with boxes: {} ms".format(1000*(t2 - t1)))
    return note_sequence


def my_test(path):
    fluidsynth.init('sfs/soundfont.sf2', 'alsa')
    print(path)
    img = read_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    bin_img = (1 * (gray == 255)).astype(np.uint8)

    staff_thickness, staff_spacing = compute_staff(gray)

    params = {}
    with open('templates/conf.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            args = line.split('=')
            params[args[0]] = int(args[1])

    required_staff_spacing = params['staff_spacing']
    
    print("Adjusting image height...")
    r = required_staff_spacing/staff_spacing
    
    adjusted_height = round(r * img.shape[0])

    img = imutils.resize(img, height=adjusted_height)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    bin_img = (1 * (gray == 255)).astype(np.uint8)

    t1 = time.time()
    staff_thickness, staff_spacing = compute_staff(gray)
    t2 = time.time()
    print("Time taken to compute staff dimensions: {} ms".format(1000*(t2 - t1)))

    t1 = time.time()
    staves = find_staves(bin_img, staff_thickness, staff_spacing)
    t2 = time.time()
    print("Time taken to find staves in image: {} ms".format(1000*(t2 - t1)))

    t1 = time.time()
    for staff in staves:
        # draw_staff(img, staff, staff_thickness, staff_spacing, (0,0,255), 1)
        bin_img = remove_staff(bin_img, staff, staff_thickness)
    t2 = time.time()
    print("Time taken to remove staves in image: {} ms".format(1000*(t2 - t1)))

    t1 = time.time()
    track_bounds = segment_by_staves(gray, staves, staff_thickness, staff_spacing)
    t2 = time.time()
    print("Time taken to segment tracks by their staves: {} ms".format(1000*(t2 - t1)))

    draw_projection_plots = 0

    note_sequence = []

    all_symbols = []

    t1 = time.time()
    for staff, track_bound in zip(staves, track_bounds):
        track = bin_img[track_bound[0]:track_bound[1]]
        
        symbols = find_all_symbols(track, staff_thickness, staff_spacing, draw_projection_plots=draw_projection_plots)
        y_offset = track_bound[0]

        for x1, y1, x2, y2 in symbols:
            all_symbols.append((x1, y1+y_offset, x2, y2+y_offset))
    
    t2 = time.time()
    print("Time taken to bound all symbols with boxes: {} ms".format(1000*(t2 - t1)))

    bw_img = (255 * bin_img).astype(np.uint8)

    t1 = time.time()
    dictionary = load_dictionary()
    t2 = time.time()
    print("Time taken to load the dictionary of templates: {} ms".format(1000*(t2 - t1)))

    t1 = time.time()
    for symbol in all_symbols:
        x1, y1, x2, y2 = symbol
        # vertical_lines = find_vertical_lines(bin_img[y1:y2, x1:x2], staff_thickness, staff_spacing)

        symbol_name, coords = match_symbol(bin_img, symbol, dictionary)
        if symbol_name != '-1':
            cv2.putText(img, symbol_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

        # for line in vertical_lines:
        #     cv2.line(img, (line[0]+x1, line[1]+y1), (line[0]+x1, line[2]+y1), (255,0,0), 2)
    
    t2 = time.time()
    print("Time taken to find all vertical lines within symbols: {} ms".format(1000*(t2 - t1)))

    cv2.imshow("Gray", gray)
    cv2.imshow("Image", img)
    cv2.imshow("Bin Image", (255 * bin_img).astype(np.uint8) )
    cv2.waitKey(205 * draw_projection_plots)
    if draw_projection_plots:
        plt.show()

    for note in note_sequence:
        play(note[0], note[1])
    cv2.destroyAllWindows()
    return staff_spacing, all_symbols, bw_img

def create_samples():
    staff_spacing, symbols, bw_img = my_test("src/samples.png")
    
    for symbol in symbols:
        x1, y1, x2, y2 = symbol
        cv2.imwrite('templates/symbol{}.png'.format(symbol), bw_img[y1:y2, x1:x2])
    
    with open('templates/conf.txt', 'w') as outf:
        outf.write("staff_spacing={}".format(staff_spacing))

def main():
    # create_samples()
    # test()
    for path in glob.glob("src/*.png"):
        if path != "src/samples.png":
            my_test(path)
    # my_test("src/samples.png")
    # my_test("src/half_note.png")
    # my_test("src/ode_to_joy.png")
    # my_test("src/bar_keysig.png")
    # my_test("src/treble_clef.png")

if __name__ == "__main__":
    main()
