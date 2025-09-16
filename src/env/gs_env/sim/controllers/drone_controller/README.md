### Run demos
- Make sure your PC has one GPU at least
- Enter your conda env
- Install dependencies by: `pip install -e`

#### Use RC control FPV in Genesis
- Flash HEX file in `./modified_BF_firmware/betaflight_4.4.0_STM32H743_forRC.hex` to your FCU (for STM32H743)
- Use Type-c to power the FCU, and connect UART port (on FCU) and USB port (on PC) through USB2TTL module
- Connect the FC and use mavlink to send FC_data from FCU to PC
- Use `ls /dev/tty*` to check the port id and modified param `USB_path` in `./config/flight.yaml`
- Do this since the default mavlink frequence for rc_channle is too low
- Use RC to control the sim drone by:
    ```
    python src/env/gs_env/sim/controllers/drone_controller/eval/rc_FPV_eval.py
    ```
#### Position controller test
- Try to get the target with no planning, thus **has poor performance**
    ```
    python src/env/gs_env/sim/controllers/drone_controller/eval/pos_ctrl_eval.py
    ```

### NOTE
- Use `pip install pymavlink` to install pymavlink
- Add `export SETUPTOOLS_USE_DISTUTILS=stdlib` into ~/.bashrc if distutils.core has been pointed to distutils but not setuptools/_distutils, or an assert will be triggered in some cases
