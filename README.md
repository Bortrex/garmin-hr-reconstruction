# VTT Apéro

## Initial state: Heart-rate data before reconstruction

During the MTB ride on October 16, 2026, the Garmin Edge 840 lost the
heart-rate broadcast from a Forerunner 265. The Edge activity consequently
contains only a short initial period of heart-rate measurements. The watch's
wellness FIT file provides additional sparse HR observations for the remainder
of the ride. These recorded observations serve as inputs for estimating the
missing heart-rate sequence.

![Recorded watch and Edge heart-rate observations and watch sampling intervals before reconstruction](docs/images/watch_wellness_hr_2026-10-16.png)

*Available watch HR observations before reconstruction, alongside the initial
Edge HR measurements. The lower panel shows intervals between stored watch
observations. The points are recorded data, not a reconstructed continuous
signal.*
