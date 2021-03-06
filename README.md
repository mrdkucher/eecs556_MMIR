## EECS 556 Final Project:
### Multimodal Image Registration: Comparison of Methods for 3D MRI to 3D Ultrasound Image Registration with Classical and Deep-Learning Accelerated Approaches

Dinank Gupta, David Kucher, Daniel Manwiller, Ellen Yeats

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
