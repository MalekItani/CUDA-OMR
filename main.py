import cv2
import numpy as np
import time
import glob
import matplotlib.pyplot as plt
import imutils
import os

def full_step(n):
    note = (n - 1) % 12
    if note == 2 or note == 7:
        return n + 1
    else:
        return n + 2

def advance(n, n_step):
    i = 0
    while i < n_step:
        n = full_step(n)
        i += 1
    return n

def classify2(note, staff, staff_thickness, staff_spacing):
    """
    note: Center of the note
    staff: array of 5 points denoting the position of the lines of the staff
    """
    note_increment = (staff_thickness + staff_spacing)//2
    n = 44
    delta_n = int(round(( (staff[-1] + staff_thickness//2) - note[1])/note_increment))
    return advance(n, delta_n)

# Start here.
def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # img = cv2.blur(img, (1, 1))
    img_shape = 400
    # img = cv2.resize(img, dsize=(img_shape, int(img_shape/img.shape[1] * img.shape[0])))
    img = imutils.resize(img, height=img_shape)
    return img

def load_dictionary(template_path='templates'):
    dictionary = {}
    for path in glob.glob('{}/*.png'.format(template_path)):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
        bin_img = (1 * (gray == 255)).astype(np.uint8)
        name = os.path.basename(path)[:-4]
        dictionary[name] = bin_img
    return dictionary

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
        # above and below the staff.
        y1 = max(staff[0] - 3*(staff_thickness + staff_spacing), 0)
        y2 = min(staff[-1] + 3*(staff_thickness + staff_spacing), img.shape[0])
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
        if proj[i] < threshold:
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
        name = 'whole_note'
    else:
        # Check if the position of the max lies near the center for the note to be flagged.
        if x_proj_argmax >= len(x_proj)//3 and x_proj_argmax <= 2*len(x_proj)//3:
            name = 'eighth_note'
        else:
            name = 'quarter_note'

    return [(name, (cx, cy))]

def _match_and_slide(I, symbol, mask, bound=False):
    """
    Given an input binary image I, a bounding rectangle symbol, and a template mask, find the
    most probably position where this template could be in this symbol by sliding this 
    template accross the symbol and counting the number of matching pixels.
    A score is assigned equal to the number of matching pixels / number of pixels in the templat 
    """
    # Initialize symbol rectangle bounds and mask rectangle bounds
    x1, y1, x2, y2 = symbol
    mask_height, mask_width = mask.shape
    nrows, ncols = I.shape

    # Initialize score and position variables
    score = 0
    pos = (-1, -1)

    # For most templates, the width and height should be a bit close to the symbol. 
    # However, for empty and filled note templates, these could be found anywhere within
    # the symbol. In order to control this behavior, we use the bound parameter.
    # If bound is set, we do the following:
    # In order to speed things up, only consider symbols whose height and width are close
    # to the height and width of the mask. This avoids trying to slide very small masks
    # over large symbols, which obviously do not match.

    # Find the ratio of symbol width to mask width and ration of symbol height to mask height
    # If these ratios are too large or too small, then don't try proceed.
    rx = (x2 - x1) / mask_width
    ry = (y2 - y1) / mask_height

    in_range = lambda x, a, b: a <= x and x <= b

    min_ratio = 0.8
    max_ratio = 1.2

    # If bound is not set, then we proceed normally:
    if not bound or (bound and in_range(rx, min_ratio, max_ratio) and in_range(ry, min_ratio, max_ratio)):
        # Loop over every pixel in the symbol to choose the top left corner (i, j) and try matching.
        # From experimentation, it helps to consider at the least the first pixel, even if
        # the bounds of the template exceed the bounds of the symbol. Usually, they don't exceed
        # exceed them by too much if there really is a match.
        for i in range(y1, max(y2 - mask_height, y1 + 1)):
            for j in range(x1, max(x2 - mask_width, x1 + 1)):
                tmp = 0
                for x in range(mask_height):
                    for y in range(mask_width): 
                        if i + x < nrows and j + y < ncols:
                            tmp += (I[i+x, j+y] == mask[x,y])
                
                if tmp > score:    
                    score = tmp
                    pos = (j + mask_width//2, i + mask_height//2) # Pos should be the exact center of where the match takes place
    
    return score/(mask_height * mask_width), pos

def _match_all(I, symbol, mask, confidence):
    """
    Same as _match_and_slide, but returns all possible locations of a template in the symbol,
    whose score exceeds a certain confidence level. This is useful when trying to find filled
    notes (which may be many) in a collection beam of beamed notes, which we can't further segment.
    """
    x1, y1, x2, y2 = symbol
    mask_height, mask_width = mask.shape
    nrows, ncols = I.shape
    pos = []
    i = y1
    # We loop over (i, j) as discussed in _match_and_slide.
    while i < max(y2 - mask_height, y1 + 1):
        # If we do find a match, then we set a flag, so that we know we should jump by one 
        # whole mask_height in the next iteration of i
        flag = 0
        j = x1
        while j < max(x2 - mask_width, x1 + 1):
            score = 0
            for x in range(mask_height):
                for y in range(mask_width):
                    if i + x < nrows and j + y < ncols:
                        score += (I[i+x, j+y] == mask[x,y])
            score /= (mask_height * mask_width)
            # Check if the score is above the confidence to add it to the list of possible
            # locations of the mask.
            if score >= confidence:
                pos.append((j + mask_width//2, i + mask_height//2))
                j += mask_width
                flag = 1
            else:
                j += 1
        if flag:
            i += mask_height
        else:
            i += 1
    return pos

def match_symbol(I, symbol, dictionary, staff_thickness, staff_spacing, filled_confidence=0.8, empty_confidence=0.7, symbol_confidence=0.6):
    """
    Given a binary image I, a bounding rectangle symbol, a collection of templates dictionary,
    try to recognize the musical character(s) found in symbol.
    Each recognition is given a score and one with the highest score is chosen, only if its score
    exceeds a certain confidence threshold.
    Returns a list of (name, pos) tuples representing the character name and character position, for
    every character found in the symbol.
    """
    # First try matching this symbol to all templates except for the filled and empty notes.
    scores = []
    for name, mask in dictionary.items():
        if name == 'filled_note' or name == 'empty_note':
            continue
        score, pos = _match_and_slide(I, symbol, mask, bound=1)
        scores.append( (score, name, pos) )
    scores.sort(reverse=True)
    # print(scores)
    if scores[0][0] >= symbol_confidence:
        return [(scores[0][1], scores[0][2])]

    # If no template matches, then try finding empty note heads in the symbol.
    score_empty, pos = _match_and_slide(I, symbol, dictionary['empty_note'])
    # print(score_empty)
    if score_empty >= empty_confidence:
        return [('half_note', pos)]

    # If no empty note head are found, then try finding filled note heads in the symbol.
    pos = _match_all(I, symbol, dictionary['filled_note'], confidence=filled_confidence)

    # If multiple filled note heads are found, then this symbol is a beamed collection of notes,
    # return each one individually without any further processing. Otherwise, then this is only
    # a single note head, which may either be a quarter note or an eigthth note, so do some more
    # processing to figure it out.
    # Note that if no notes are detected, then this symbol doesn't represent anything important,
    # and an empty list will be returned. 
    if len(pos) == 1:
        if mask.shape[1] / (symbol[2] - symbol[0]) < 0.9:
            return [('quarter_note', pos[0])]
        else:
            return [('eighth_note', pos[0])]    
        # return recognize_isolated_note(I, symbol, staff_thickness, staff_spacing)
    else:
        return [('eighth_note', c) for c in pos]

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
    # its length is <= staff_thickness + 3
    for x in staff:
        x += 1
        for j in range(ncols):
            if Iv[x, j] ==  0:
                continue

            x2 = x
            while x2 < nrows and Iv[x2, j] > 0:
                x2 += 1
            
            if Iv[x2-1, j] <= staff_thickness + 3:
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

template_path = 'templates'
dictionary = load_dictionary(template_path)

def my_test(path):
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

    all_symbols = []

    t1 = time.time()
    for track_id, (staff, track_bound) in enumerate(zip(staves, track_bounds)):
        track = bin_img[track_bound[0]:track_bound[1]]
        
        symbols = find_all_symbols(track, staff_thickness, staff_spacing, draw_projection_plots=draw_projection_plots)
        y_offset = track_bound[0]
        
        for x1, y1, x2, y2 in symbols:
            all_symbols.append( ((x1, y1+y_offset, x2, y2+y_offset), track_id) )
    
    t2 = time.time()
    print("Time taken to bound all symbols with boxes: {} ms".format(1000*(t2 - t1)))

    note_sequences = [[] for i in range(len(staves))]

    t1 = time.time()
    for i, (symbol, track_id) in enumerate(all_symbols):
        x1, y1, x2, y2 = symbol
        print("Recognizing symbol {} ...".format(i))
        characters = match_symbol(bin_img, symbol, dictionary, staff_thickness, staff_spacing)

        for (name, pos) in characters:
            cv2.putText(img, name, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255))
            note_sequences[track_id].append((name, pos))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    
    t2 = time.time()
    print("Time taken to find classify all symbols: {} ms".format(1000*(t2 - t1)))

    print(note_sequences)
    with open('tests/out.txt', 'w') as f:
        for i in range(len(note_sequences)):
            f.write(f'Track{i}: ')
            for name, pos in note_sequences[i]:
                if name.endswith('note'):
                    name = name + '.{}'.format(classify2(pos, staves[i], staff_thickness, staff_spacing))
                f.write(f'{name} ')
            f.write('\n')

    cv2.imshow("Gray", gray)
    cv2.imshow("Image", img)
    cv2.imshow("Bin Image", (255 * bin_img).astype(np.uint8) )
    cv2.waitKey(205 * draw_projection_plots)
    if draw_projection_plots:
        plt.show()

    cv2.destroyAllWindows()

def create_samples():
    img = read_image('src/samples.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    bin_img = (1 * (gray == 255)).astype(np.uint8)
    
    staff_thickness, staff_spacing = compute_staff(gray)
    staves = find_staves(bin_img, staff_thickness, staff_spacing)

    for staff in staves:
        # draw_staff(img, staff, staff_thickness, staff_spacing, (0,0,255), 1)
        bin_img = remove_staff(bin_img, staff, staff_thickness)

    track_bounds = segment_by_staves(gray, staves, staff_thickness, staff_spacing)

    all_symbols = []

    for staff, track_bound in zip(staves, track_bounds):
        track = bin_img[track_bound[0]:track_bound[1]]
        
        symbols = find_all_symbols(track, staff_thickness, staff_spacing)
        y_offset = track_bound[0]

        for x1, y1, x2, y2 in symbols:
            all_symbols.append((x1, y1+y_offset, x2, y2+y_offset))

    for symbol in all_symbols:
        x1, y1, x2, y2 = symbol
        sym = (255 * bin_img[y1:y2, x1:x2]).astype(np.uint8)
        cv2.imshow('symbol', sym)
        cv2.waitKey(100)
        name = input('What do you want to call this? ')
        if name == 'skip':
            continue
        else:
            cv2.imwrite('{}/{}.png'.format(template_path, name), sym, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    with open('{}/conf.txt'.format(template_path), 'w') as outf:
        outf.write("staff_spacing={}".format(staff_spacing))

    cv2.imshow("Image", img)
    cv2.imshow("Bin Image", (255 * bin_img).astype(np.uint8) )
    cv2.waitKey(0)

def main():
    # create_samples()
    # test()
    # for path in glob.glob("src/*.png"):
    #     my_test(path)
    # my_test("src/samples.png")
    # my_test("src/half_note.png")
    # my_test("src/ode_to_joy.png")
    my_test("src/bar_keysig.png")
    # my_test("src/bass_clef.png")
    # my_test("src/three_bar.png")

if __name__ == "__main__":
    main()
