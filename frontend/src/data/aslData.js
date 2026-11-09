/**
 * Authentic ASL Alphabet Reference Data (A-Z)
 * Includes finger positions, thumb orientation, descriptive instructions,
 * and SVG icon representations for each fingerspelled letter.
 */

export const ASL_ALPHABET = [
  {
    letter: 'A',
    name: 'Fist with thumb up',
    description: 'Make a fist with all four fingers curled tightly into the palm. Place the thumb resting upright flat against the side of the index finger.',
    confusions: 'Often confused with S (where thumb crosses in front) or E (where fingertips rest on thumb).',
    category: 'Fist',
    difficulty: 'Easy'
  },
  {
    letter: 'B',
    name: 'Open flat hand',
    description: 'Hold all four fingers straight up, pressed closely together. Fold the thumb flat across the palm.',
    confusions: 'Keep the 4 fingers pressed together, not spread (which is 4 or 5).',
    category: 'Open Palm',
    difficulty: 'Easy'
  },
  {
    letter: 'C',
    name: 'Curved C cup',
    description: 'Curve all fingers and the thumb in an arc to form the shape of the letter "C", viewed sideways.',
    confusions: 'Ensure the hand curves gently without closing into an O.',
    category: 'Form / Curved',
    difficulty: 'Easy'
  },
  {
    letter: 'D',
    name: 'Index pointing up',
    description: 'Point your index finger straight up. Curl middle, ring, and pinky fingers down so their tips touch the tip of the thumb, forming a circle.',
    confusions: 'Often confused with F (where index touches thumb and 3 fingers point up).',
    category: 'Pointing',
    difficulty: 'Medium'
  },
  {
    letter: 'E',
    name: 'Curled claw on thumb',
    description: 'Curl all four fingers downward tightly with tips resting on top of the thumb folded beneath them.',
    confusions: 'Fingertips must rest on or touch the side of the tucked thumb.',
    category: 'Fist',
    difficulty: 'Medium'
  },
  {
    letter: 'F',
    name: 'OK sign / 3 fingers up',
    description: 'Touch the tip of your index finger to the tip of your thumb to form a circle. Extend the middle, ring, and pinky fingers straight up and slightly spread.',
    confusions: 'The opposite of D. In F, 3 fingers are up; in D, only 1 finger is up.',
    category: 'Pointing / Ring',
    difficulty: 'Easy'
  },
  {
    letter: 'G',
    name: 'Horizontal pinch',
    description: 'Point index finger horizontally to the left (or forward). Thumb points parallel along the index finger, like measuring a small gap.',
    confusions: 'Often confused with H (which has 2 fingers extended).',
    category: 'Directional',
    difficulty: 'Medium'
  },
  {
    letter: 'H',
    name: 'Two fingers horizontal',
    description: 'Extend both the index and middle fingers horizontally pressed together. Fold the ring and pinky fingers into the palm with thumb resting over them.',
    confusions: 'Like G, but with two fingers instead of one.',
    category: 'Directional',
    difficulty: 'Medium'
  },
  {
    letter: 'I',
    name: 'Pinky finger up',
    description: 'Extend the pinky finger straight up. Fold the index, middle, and ring fingers into the palm with the thumb resting across them.',
    confusions: 'Often confused with J (which traces an arc) or Y (which also extends the thumb).',
    category: 'Single Finger',
    difficulty: 'Easy'
  },
  {
    letter: 'J',
    name: 'Pinky hook trace',
    description: 'Start with the pinky finger extended (like I), then trace a downward "J" curve in the air, rotating the wrist inwards.',
    confusions: 'Dynamic sign starting from the I handshape.',
    category: 'Dynamic',
    difficulty: 'Medium'
  },
  {
    letter: 'K',
    name: 'V with thumb between',
    description: 'Point index finger straight up, extend middle finger forward/up at an angle, and place the thumb knuckle between them.',
    confusions: 'Similar to V and P (P points downwards).',
    category: 'Special',
    difficulty: 'Hard'
  },
  {
    letter: 'L',
    name: 'L-shape angle',
    description: 'Extend index finger straight up and thumb straight out to the side at a 90-degree angle, forming the capital letter "L".',
    confusions: 'One of the most recognizable ASL signs.',
    category: 'Form',
    difficulty: 'Easy'
  },
  {
    letter: 'M',
    name: 'Three fingers over thumb',
    description: 'Fold index, middle, and ring fingers over your tucked thumb so the thumb tip pokes out between ring and pinky fingers.',
    confusions: 'Distinguished from N (2 fingers) and T (1 finger) by having 3 fingers over thumb.',
    category: 'Fist Fold',
    difficulty: 'Hard'
  },
  {
    letter: 'N',
    name: 'Two fingers over thumb',
    description: 'Fold index and middle fingers over your tucked thumb so the thumb tip pokes out between middle and ring fingers.',
    confusions: 'Distinguished from M by folding 2 fingers rather than 3.',
    category: 'Fist Fold',
    difficulty: 'Hard'
  },
  {
    letter: 'O',
    name: 'O-ring circle',
    description: 'Curve all fingers and thumb together so all fingertips touch the tip of the thumb, creating an "O" shape.',
    confusions: 'Ensure all 4 fingers touch the thumb, unlike C which stays open.',
    category: 'Form / Curved',
    difficulty: 'Easy'
  },
  {
    letter: 'P',
    name: 'Downward K',
    description: 'Form the K handshape (index up, middle angled, thumb between) but point the entire hand downwards with wrist angled.',
    confusions: 'P is simply an inverted K.',
    category: 'Directional',
    difficulty: 'Hard'
  },
  {
    letter: 'Q',
    name: 'Downward G',
    description: 'Form the G handshape (index and thumb parallel) but point the fingers straight downwards.',
    confusions: 'Q is simply an inverted G.',
    category: 'Directional',
    difficulty: 'Hard'
  },
  {
    letter: 'R',
    name: 'Crossed fingers (Good luck)',
    description: 'Extend the index and middle fingers straight up and cross the middle finger over the index finger.',
    confusions: 'Traditional "fingers crossed for luck" gesture.',
    category: 'Crossed',
    difficulty: 'Easy'
  },
  {
    letter: 'S',
    name: 'Fist with thumb in front',
    description: 'Make a tight fist with all four fingers curled, and wrap the thumb firmly across the front of the fingers.',
    confusions: 'Confused with A (thumb is at side in A, across front in S).',
    category: 'Fist',
    difficulty: 'Medium'
  },
  {
    letter: 'T',
    name: 'Thumb between index and middle',
    description: 'Make a fist with the thumb poking up between the curled index finger and middle finger.',
    confusions: 'Part of the M-N-T group. T has 1 finger over thumb.',
    category: 'Fist Fold',
    difficulty: 'Medium'
  },
  {
    letter: 'U',
    name: 'Two fingers up together',
    description: 'Hold index and middle fingers straight up, pressed tightly against each other. Thumb folds across palm.',
    confusions: 'Distinguished from V (fingers spread) and H (fingers horizontal).',
    category: 'Pointing',
    difficulty: 'Easy'
  },
  {
    letter: 'V',
    name: 'Peace sign / V-spread',
    description: 'Hold index and middle fingers straight up and spread apart in a "V" shape. Thumb folds across palm.',
    confusions: 'Fingers must be separated, unlike U where they touch.',
    category: 'Pointing',
    difficulty: 'Easy'
  },
  {
    letter: 'W',
    name: 'Three fingers up spread (W)',
    description: 'Extend index, middle, and ring fingers straight up and spread apart. Thumb holds pinky finger down against palm.',
    confusions: 'Three fingers form the three prongs of the letter "W".',
    category: 'Pointing',
    difficulty: 'Easy'
  },
  {
    letter: 'X',
    name: 'Index bent into hook',
    description: 'Fold middle, ring, and pinky into the palm. Bend index finger at the second knuckle into a curved hook.',
    confusions: 'Like a pirate hook or clicking gesture.',
    category: 'Hook',
    difficulty: 'Medium'
  },
  {
    letter: 'Y',
    name: 'Hang loose / Shaka sign',
    description: 'Extend thumb and pinky finger outward while folding middle three fingers (index, middle, ring) into the palm.',
    confusions: 'Identical to the universal "hang loose" or phone sign.',
    category: 'Spread',
    difficulty: 'Easy'
  },
  {
    letter: 'Z',
    name: 'Index tracing Z',
    description: 'Extend index finger and draw the letter "Z" in the air (horizontal right, diagonal down-left, horizontal right).',
    confusions: 'Dynamic gesture using index finger.',
    category: 'Dynamic',
    difficulty: 'Medium'
  }
];

export const ASL_CATEGORIES = ['All', 'Fist', 'Pointing', 'Open Palm', 'Form / Curved', 'Directional', 'Crossed', 'Dynamic', 'Fist Fold'];
