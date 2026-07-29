""""

# Calculate movement between two images.
dx = x2 - x1
dy = y2 - y1
d = sqrt(dx*dx + dy*dy)

# Ignore jitter. If movement exceeds the jitter threshold, subtract the deadband.
if d < jitterThreshold:
    d = 0
else:
    d = d - jitterThreshold

# Accumulate distance in the rolling five-minute window.
rollingDistance += d
rollingDistance -= distanceFrom300SecondsAgo

# Determine the sleep state.
if rollingDistance > awakeThreshold:
    state = AWAKE
else:
    state = ASLEEP

"""