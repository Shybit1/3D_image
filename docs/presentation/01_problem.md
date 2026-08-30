# 01 — Problem

**SIH26175 (ISRO): DepthWizard — Single-View Height Estimation and 3D Flythrough**

## The Core Challenge

Reconstructing 3D structure from imagery normally requires two or more
views of the same scene (stereo pairs, multi-date tasking, or LiDAR).
Most single-date optical satellite archives — the overwhelming majority of
what's actually available for a given place at a given time — are single
views. If we can extract usable height information from a *single* image,
we unlock 3D analysis for imagery that would otherwise only ever be a flat
picture.

## What ISRO Is Asking For

1. Take one optical satellite image (no stereo pair required).
2. Estimate relative height/depth using a pretrained monocular depth
   model (not a from-scratch novel network — explicit in the problem
   statement).
3. Where the image is georeferenced, convert relative depth into absolute
   elevation using auxiliary data: a coarse DEM, SRTM, or a handful of
   Ground Control Points.
4. Turn that elevation surface into a navigable, textured 3D environment.
5. Prove it's not just a pretty picture: report DSM accuracy (RMSE/MAE/
   correlation) against real reference elevation, checked for stability
   across different terrain types (urban, sparse, hilly, forested).

## Why This Is Hard (Not a Solved Problem)

A single image is fundamentally scale-ambiguous — there's no built-in way
to know if you're looking at a 5m building from far away or a 50m building
from farther away, without extra information. Monocular depth models solve
a *related* but different problem (relative depth ordering in natural
photos), and — as we found and measured directly in this build — they
carry priors from their training data (photos taken by people standing on
the ground) that do not transfer cleanly to a satellite's overhead view.
Bridging that gap honestly, rather than hiding it, is the actual technical
problem.
