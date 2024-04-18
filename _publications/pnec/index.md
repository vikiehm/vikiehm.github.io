---
layout: page
permalink: publications/pnec/
date: 2022_05_26 # determines sorting just take the date of the first publication as YYYY_MM_DD
image: assets/teaser.pdf
image_mouseover: assets/trajectory_8.pdf

title: "Probabilistic Normal Epipolar Constraint for Frame-To-Frame Rotation Optimization under Uncertain Feature Positions"
venue: CVPR, 2022
authors:
  - name: dominikmuhle
    affiliations: "1"
    equal_contribution: True
  - name: lukaskoestler
    affiliations: "1"
    equal_contribution: True
  - name: nikodemmel
    affiliations: "1"
  - name: florianbernard
    affiliations: "1,2"
  - name: danielcremers
    affiliations: "1"
  
affiliations:
  - name: tum
    length: short
  - name: bonn
    length: short
  
description: "We propose a probabilistic extension to the normal epipolar constraint (NEC) which we call the PNEC. It allows to account for keypoint position uncertainty in images to produce more accurate frame to frame pose estimates. "

links:
    - name: Project Page
      link: publications/pnec/
    - name: Paper
      link: https://arxiv.org/abs/2204.02256
      style: "bi bi-file-earmark-richtext"
    - name: Code
      link: https://github.com/tum-vision/pnec
      style: "bi bi-github"
    - name: Video
      link: https://youtu.be/YraCHnR5dmg
      style: "bi bi-youtube"
citation: '@inproceedings{muhle2022pnec, 
  author = {D Muhle and L Koestler and N Demmel and F Bernard and D Cremers},
  title = {The Probabilistic Normal Epipolar Constraint for Frame-To-Frame Rotation Optimization under Uncertain Feature Positions}, 
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022},
}'
acknowledgements: 'We thank our colleagues, especially Florian Hofherr, for proofreading and helpful discussions. This work was supported by the ERC Advanced Grant SIMULACRON and by the Munich Center for Machine Learning.'

---

<section class="hero teaser">
  <div class="container is-max-desktop">
    <div class="hero-body">
      <img src="assets/teaser.pdf" width="100%" />
    </div>
  </div>
</section>

<section class="section hero is-light">
  <div class="container is-max-desktop">
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <h2 class="title is-3">Abstract</h2>
        <div class="content has-text-justified">
          <p>
              The estimation of the relative pose of two camera views is a fundamental problem in computer vision. Kneip et al. proposed to solve this problem by introducing the NEC. However, their approach does not take into account uncertainties, so that the accuracy of the estimated relative pose is highly dependent on accurate feature positions in the target frame. In this work, we introduce the PNEC that overcomes this limitation by accounting for anisotropic and inhomogeneous uncertainties in the feature positions. To this end, we propose a novel objective function, along with an efficient optimization scheme that effectively minimizes our objective while maintaining real-time performance. In experiments on synthetic data, we demonstrate that the novel PNEC yields more accurate rotation estimates than the original NEC and several popular relative rotation estimation algorithms. Furthermore, we integrate the proposed method into a state-of-the-art monocular rotation-only odometry system and achieve consistently improved results for the real-world KITTI dataset.
          </p>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="hero is-small">
  <div class="hero-body">
    <div class="container">
      <div class="columns is-centered has-text-centered">
        <div class="column is-four-fifths">
          <h2 class="title is-3 has-text-left">Incooporating Uncertainty into Pose Estimation</h2>
          <p class="content has-text-justified">
            The NEC allows for relative pose estimation by enforcing the coplanarity of \emph{epipolar plane normal vectors} constructed from feature correspondences. An energy function to minimize the NEC can be written as:
            $$
              E(\boldsymbol{R}, \boldsymbol{t}) = \sum_i e_i^2 = \sum_i | \boldsymbol{t}^\top (\boldsymbol{f}_i \times \boldsymbol{R} \boldsymbol{f}^\prime_i) |^2
            $$
            The PNEC extends the NEC by incorporating uncertainty. To be more specific, the PNEC allows the use of the anisotropic and inhomogeneous nature of the uncertainty of the feature position in the energy function. We assume that the position error follows a 2D Gaussian distribution in the image plane with a known covariance matrix $\boldsymbol{\Sigma}_{2\text{D},i}$ per feature.
            <br>
            Given the 2D covariance matrix of the feature position in the target frame $\boldsymbol{\Sigma}_{2\text{D},i}$, we propagate it through the unprojection function using the unscented transform in order to obtain the 3D covariance matrix $\boldsymbol{\Sigma}_i$ of the bearing vector $\boldsymbol{f}^\prime_i$.
            Using the unscented transform ensures full-rank covariance matrices after the transform. We derive the details of the unscented transform in the supplementary material and show qualitative examples. Propagating this distribution to the normalized epipolar error gives the probabilistic distribution of the residual. Due to the linearity of the transformations, the distribution of the residual is a univariate Gaussian distribution $\mathcal{N}(0, \sigma_i^2)$, with variance
            $$
              \sigma_i^2(\boldsymbol{R}, \boldsymbol{t}) = \boldsymbol{t}^\top \hat{\boldsymbol{f}_i} \boldsymbol{R} \boldsymbol{\Sigma}_i \boldsymbol{R}^\top \hat{\boldsymbol{f}_i}{}^\top \boldsymbol{t}
            $$
            $$
              E_P(\boldsymbol{R}, \boldsymbol{t}) = \sum_i \frac{e_i^2}{\sigma_i^2} = \sum_i \frac{| \boldsymbol{t}^\top (\boldsymbol{f}_i \times \boldsymbol{R} \boldsymbol{f}^\prime_i) |^2}{\boldsymbol{t}^\top \hat{\boldsymbol{f}_i} \boldsymbol{R} \boldsymbol{\Sigma}_i \boldsymbol{R}^\top \hat{\boldsymbol{f}_i}{}^\top \boldsymbol{t}}
            $$
          </p>
          <h2 class="title is-3 has-text-left">Sythetic Experiments</h2>
          <p class="content has-text-justified">
            With the simulated experiments we evaluate the performance of the PNEC in a frame-to-frame setting. The experiments consist of randomly generated problems of two frames with known correspondences. The simulated experiments show the benefit of incorporating uncertainty into the optimization.
          </p>
          <p float="left">
            <img src="assets/omni_mean_rot.pdf" width="300" />
            <img src="assets/omni_mean_t.pdf" width="300" />
          </p>
          <p class="content has-text-justified">
            <i><b>Omnidirectional Cameras.</b> Rotation and translation error for anisotropic and inhomogeneous noise</i>
            <br><br>
            The figures shows the rotational and translational error for experiment with omni-directional camera setups for anisotropic inhomogeneous noise for both experiments. The PNEC achieves consistently better results for the rotation over all noise levels.
          </p>
          <p float="left">
            <img src="assets/pin_mean_rot.pdf" width="300" />
            <img src="assets/pin_mean_t.pdf" width="300" />
          </p>
          <p class="content has-text-justified">
            <i><b>Pinhole Cameras.</b> Rotation and translation error for anisotropic and inhomogeneous noise</i>
            <br><br>
            The figures shows the rotational and translational error for experiment with pinhole cameras camera setups for anisotropic inhomogeneous noise for both experiments. The PNEC achieves consistently better results for the rotation over all noise levels.
            <br>
            Please refer to the paper and supplementary material for a more detailed evaluation on synthetic experiments.
          </p>
          <h2 class="title is-3 has-text-left">KITTI Evaluation</h2>
          <p float="left">
            <img src="assets/trajectory_7.pdf" width="360" />
            <img src="assets/trajectory_8.pdf" width="360" />
          </p>
          <p class="content has-text-justified">
            <i><b>Trajetory.</b> KITTI seq. 07 and 08. Visualization uses the ground truth translation as we do monocular pose estimation and focus on rotation estimation.</i>
            <br><br>
            The PNEC also improves on non-probabilistic monocular visual odometry methods that use the NEC. Please refer to the paper for a more detailed evaluation of the PNEC on the visual odometry task on KITTI.
          </p>
        </div>
      </div>
    </div>
  </div>
</section>