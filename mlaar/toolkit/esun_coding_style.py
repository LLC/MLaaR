
# coding: utf-8


"""To run and show the pylint report.
"""
import os
import re
import json
import linecache
import logging
from subprocess import PIPE
from subprocess import Popen
import numpy as np
import pandas as pd
from IPython import display


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


try:
    PKG_PATH = os.path.abspath(os.path.dirname(__file__))
except NameError:
    logging.warning('name __file__ is not defined.')
    PKG_PATH = '.'
RULE_PATH = os.path.join(PKG_PATH, 'rule/pylint_rules.csv')
RC_PATH = os.path.join(PKG_PATH, 'config/pylintrc')


def remove_nbtag(py_script, show_remove=False, remove_check_cmd=True):
    """To remove the jupyuter notebook tag from a python script maked from a jupyter notebook.

    Args:
        py_script: The python script will be fixed.
        show_remove: Show the removed line.
        remove_check_cmd: Boolean.
            To remove commands report_coding_style() and check_coding_style() from the python script. Default is True.

    """
    # check the file exist
    assert os.path.isfile(py_script), '%s does not exist.' % py_script

    # regular expression pattern
    nb_pattern = '^# In[[0-9 ]*]:$'
    pylint_pattern = '^((\w+\.)*check_coding_style)|((\w+\.)*report_coding_style)|(import esun_coding_style)'

    py_script_tmp = py_script + '.tmp'
    with open(py_script_tmp, 'w') as f_w:
        with open(py_script, 'r') as f_r:
            for line in f_r:
                nb_match = re.match(nb_pattern, line)
                if remove_check_cmd:
                    pylint_match = re.match(pylint_pattern, line)
                else:
                    pylint_match = False
                if not nb_match and not pylint_match:
                    f_w.write(line)
                elif show_remove:
                    print(line)

    py_script_name = py_script.split('/')[-1]

    # 如果檔案存在則備份
    if os.path.isfile(py_script):
        tmp = py_script.split('/')
        py_script_name = os.path.splitext(tmp[-1])
        py_script_dir = '/'.join(tmp[:-1])
        py_script_dir = '.' if not py_script_dir else py_script_dir
        backup_name = '%s/%s_backup%s' % (py_script_dir, py_script_name[0], py_script_name[1])
        os.rename(py_script, backup_name)
    os.rename(py_script_tmp, py_script)


def run_nbconvert(notebook_name):
    """To convert a jupyter notebook to a python script.

    Args:
        notebook_name: The jupyter notebook file will be convert to the python script.

    Returns:
        Exit code of the subprocess.
    """
    # check the file exist
    assert os.path.isfile(notebook_name), '%s does not exist.' % notebook_name

    nbconvert_cmd = 'jupyter nbconvert --to script "%s"' % notebook_name
    run_cmd = Popen(nbconvert_cmd, shell=True, stdout=PIPE, stderr=PIPE)
    run_cmd.wait()
    rc = run_cmd.returncode
    if rc != 0:
        #print('Convert jupyter notebook error! Return code: %s' % rc)
        logging.info('Convert jupyter notebook error! Return code: %d', rc)
        for line in run_cmd.stderr:
            print(line.decode("utf-8"))
    return rc
# run_nbconvert('test_pylint.ipynb')


def run_autopep8(checked_program):
    """To convert a jupyter notebook to a python script.

    Reference: https://pypi.org/project/autopep8/#use-as-a-module

    Args:
        checked_program:  The program will be fixed by autopep8.

    Returns:
        Exit code of the subprocess.
    """
    # check the file exist
    assert os.path.isfile(
        checked_program), '%s does not exist.' % checked_program

    items = 'E2,E305,E306,W293,W391,W292,E303,E501,E502,E304,W292,E401,E26,E265,E231,W291'
    autopep8_cmd = 'autopep8 --in-place --aggressive --select=%s --max-line-length 120 %s' % (
        items, checked_program)
    run_cmd = Popen(autopep8_cmd, shell=True, stdout=PIPE, stderr=PIPE)
    run_cmd.wait()
    rc = run_cmd.returncode
    if rc != 0:
        #print('Run autopep8 error! Return code: %s' % rc)
        logging.info('Run autopep8 error! Return code: %d', rc)

        for line in run_cmd.stderr:
            print(line.decode("utf-8"))
    return rc


def run_pylint(py_script, show_cmd=True, rc_path=RC_PATH):
    """To run pylint report.

    Reference: http://pylint.pycqa.org/en/latest/index.html

    Args:
        py_script: string. The program will be reviewed coding style. Only support python script.
        show_cmd: True/False. To show The pylint command, default is True.

    Returns:
        A dataframe of the pylint report.
    """
    # check the file exist
    assert os.path.isfile(py_script), '%s does not exist.' % py_script

    # set pylint command
    output_format = 'json'
    #pylint_cmd = 'pylint --reports=no --disable=I --output-format=%s %s' % (output_format, py_script)
    pylint_cmd = 'pylint --reports=no --disable=I --rcfile=%s --output-format=%s %s' % (
        rc_path, output_format, py_script)

    # run report
    if show_cmd:
        # print(pylint_cmd)
        logging.info(pylint_cmd)

    run_cmd = Popen(pylint_cmd, shell=True, stdout=PIPE, stderr=PIPE)
    run_stdout, run_stderr = run_cmd.communicate()
    #print('Run pylint completely.')
    logging.info('Run pylint completely.')

    # check return code
    rc = run_cmd.returncode
    if rc in [32, 1]:
        for line in run_stderr.decode("utf-8"):
            print(line, end='')
        raise RuntimeError('Run pylint error! Return code: %r' % rc)
    elif rc != 0:
        #print('Pylint report some suggestion! Return code: %s' % rc)
        logging.info('Pylint report some suggestion! Return code: %s', rc)

    # transfer report to json
    context = []
    for line in run_stdout.decode("utf-8"):
        context.append(line)

    context = json.loads(''.join(context))

    col_names = ['type', 'module', 'obj', 'line', 'column', 'path', 'symbol', 'message', 'message-id']
    pylint_report = pd.DataFrame(dict((x, []) for x in col_names))

    # combine the report to a dataframe
    for i in list(context):
        tmp = pd.DataFrame.from_dict(i, orient='index').T
        pylint_report = pd.concat([pylint_report, tmp], axis=0)

    return_template = ['message-id', 'path', 'line', 'symbol', 'message', 'obj']
    pylint_report = pylint_report[return_template]

    return pylint_report


def run_pylint_score(py_script):
    """To run pylint report.

    Args:
        py_script: string. The program will be reviewed coding style. Only support python script.

    Returns:
        evaluation statement.
    """
    assert os.path.isfile(py_script), '%s does not exist.' % py_script

    # set pylint command
    output_format = 'text'
    pylint_cmd = 'pylint --reports=no --disable=I --output-format=%s %s' % (output_format, py_script)

    run_cmd = Popen(pylint_cmd, shell=True, stdout=PIPE, stderr=PIPE)
    run_stdout, _ = run_cmd.communicate()
    logging.info('Run pylint score completely.')

    text_report = []
    pattern = 'Your code has been rated at *.*/10 '
    for line in run_stdout.decode("utf-8"):
        text_report.append(line)

    pylint_msg = ''.join(text_report)
    get_score = re.findall(pattern, pylint_msg)
    if get_score:
        score = get_score[0]
    else:
        score = "{pgm} has error and can't be executed".format(pgm=py_script)

    return score


def check_coding_style(checked_program, convert_to_py=True, rm_nb_tag=True, remove_check_cmd=True,
                       using_autopep8=True, run_score=False, rule_path=RULE_PATH):
    """To run pylint report.

    Args:
        checked_program: The program will be reviewed coding style.
            Support both python script (.py) and jupyter notebook (.ipynb).
            If input jupyter notebook (.ipynb), must to set "convert_to_py=True".
        convert_to_py: Boolean.
            Convert python script (.py) to jupyter notebook (.ipynb). Default True.
        rm_nb_tag: Boolean.
            To remove the jupyuter notebook tag from a python script maked from a jupyter notebook. Default True.
        remove_check_cmd: Boolean.
            To remove commands report_coding_style() and check_coding_style() from the python script.
        rm_nb_tag: Boolean.
            Using autopep8 to fix the script. Default True.

    Returns:
        A dataframe of the pylint report and highlight the .
    """

    # check the extension of the file (.py or .ipynb)
    file_name, file_extension = os.path.splitext(checked_program)
    if convert_to_py and file_extension == '.ipynb':
        py_scipt = file_name + '.py'
        logging.info('Convert %s to %s' % (checked_program, py_scipt))
        run_nbconvert(checked_program)
    else:
        py_scipt = checked_program

    # remove jupyter notebook's tag
    if rm_nb_tag:
        logging.info('Remove jupyter notebook\'s tag.')
        remove_nbtag(py_scipt, remove_check_cmd=remove_check_cmd)

    # using autopep8
    if using_autopep8:
        logging.info('Using autopep8.')
        run_autopep8(py_scipt)

    py_scipt_lines = len(linecache.getlines(py_scipt))
    linecache.clearcache()
    logging.info('{script_name} has {lines} lines.'.format(
        script_name=py_scipt, lines=py_scipt_lines))

    # Running pylint
    logging.info('Running pylint.')
    pylint_report = run_pylint(py_scipt)

    if run_score:
        score = run_pylint_score(py_scipt)
    else:
        score = None

    # combine E.Sun codin style
    rules = pd.read_csv(rule_path, encoding='utf-8')
    rules.columns = ['type',
                     'message-id',
                     'message-class',
                     'message-desc',
                     'detail',
                     'required']

    cols = ['required', 'type', 'message-id']
    rules = rules[cols]
    rules.required = np.vectorize(
        lambda x: 'y' if x == 'Y' else '')(rules.required)

    output = rules.merge(pylint_report, how='inner', on=['message-id'])
    output = output.sort_values(
        by=['required', 'type', 'line', 'message-id'], ascending=[False, True, True, True])
    output.reset_index(drop=True, inplace=True)

    # statistic for report
    statistic = None
    output['count'] = 1
    statistic = output.groupby(
        by=['required', 'type']).agg({'count': 'count'})
    statistic.reset_index(inplace=True)
    statistic.sort_values(by=['required', 'type'], ascending=[False, True], inplace=True)

    output.drop(columns=['type', 'count'], axis=1, inplace=True)

    return output, statistic, score

#output = esun_coding_style_checker('pylint_test_class-R.py')


def report_coding_style(checked_program, convert_to_py=True, run_score=True):
    """To run pylint report.

    Args:
        checked_program: The program will be reviewed coding style.
            Support both python script (.py) and jupyter notebook (.ipynb).
            If input jupyter notebook (.ipynb), must to set "convert_to_py=True".
        convert_to_py: Convert python script (.py) to jupyter notebook (.ipynb).

    Returns:
        A dataframe of the pylint report and highlight the .
    """
    output, statistic, score = check_coding_style(checked_program, convert_to_py=convert_to_py, run_score=run_score)

    print('Description of Types:')
    print('[R]efactor for a "good practice" metric violation.')
    print('[C]onvention for coding standard violation.')
    print('[W]arning for stylistic problems, or minor programming issues.')
    print('[E]rror for important programming issues (i.e. most probably bug).')
    print('[F]atal for errors which prevented further processing.')
    print('')
    if score:
        print(score)
    print('')
    print('Summary of the pylint report:')
    display.display(statistic, display_id='pylint_statistic')
    print('')
    print('List the pylint items:')
    display.display(output, display_id='pylint_output')
