## EECS 556 Final Project:
### Multimodal Image Registration: Comparison of Methods for 3D MRI to 3D Ultrasound Image Registration with Classical and Deep-Learning Accelerated Approaches

Dinank Gupta, David Kucher, Daniel Manwiller, Ellen Yeats

### Running LC2 Code:
1) First install deepreg locally. Ensure you're in dir 'DeepReg' and run:
    ```bash
    pip install -e . --no-cache-dir
    ```
2) Install Py-BOBYQA
    ```bash
    pip install Py-BOBYQA
    ```
3) Run LC2 with:
    ```bash
    python demos/lc2_paired_mrus_brain/register.py -f train/fixed_images/Case1.nii.gz -m train/moving_images/Case1.nii.gz -s 32 32 36
    ```
### Python script for quickly separating the .tag file into a .txt

# Running
    ```bash
    python landmarks_split_txt.py --inputtag *folder*/Case1-MRI-beforeUS.tag --savetxt Case1_lm
    ```
    
### Git Steps:
Setup:
- `git clone https://github.com/mrdkucher/eecs556_MMIR.git`

Making changes:
- Add all changes to be committed: `git add .` or `git add <filename 1> <filename 2> ... <filename N>`
- Commit local changes to your local repo with message: `git commit -m "<commit message>"`
- Rebase any changes from remote: `git pull --rebase origin master`
  - If there are merge conflicts, resolve them by keeping whatever code should stay in
  - continue rebase by running: `git add .` and `git rebase --continue`
  - at end of rebase, you'll be prompted to update the commit message, which you can leave alone.
- Push local changes to remote branch: `git push -u origin master`, or just `git push` after you've done the former command once.

In summary:
- Make changes
- `git add .`
- `git commit -m "<message>"`
- `git pull --rebase origin master`
- `git push`
