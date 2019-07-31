# From original SQLNet code.
# Wonseok modified. 20180607

import records
import re
from babel.numbers import parse_decimal, NumberFormatError


schema_re = re.compile(r'\((.+)\)') # group (.......) dfdf (.... )group
num_re = re.compile(r'[-+]?\d*\.\d+|\d+') # ? zero or one time appear of preceding character, * zero or several time appear of preceding character.
# Catch something like -34.34, .4543,
# | is 'or'

agg_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM"}
cond_op_dict = {0:">", 1:"<", 2:"==", 3:"!="}
cond_rela_dict = {0:"and",1:"or",-1:""}

class DBEngine:

    def __init__(self, fdb):
        #fdb = 'data/test.db'
        self.db = records.Database('sqlite:///{}'.format(fdb))
        self.conn = self.db.get_connection()

    def execute_query(self, table_id, query, *args, **kwargs):
        select_index = []
        select_index.append(query.sel_index)
        aggregation_index = []
        aggregation_index.append(query.agg_index)
        return self.execute(table_id, select_index, aggregation_index, query.conditions,
                            query.condition_relation)

    def execute(self, table_id, select_index, aggregation_index, conditions, condition_relation, lower=True):
        '''
        自己吧上面两个变量变成了list
        '''
        if not table_id.startswith('Table'):
            table_id = 'Table_{}'.format(table_id.replace('-', '_'))
        wr = ""
        if condition_relation == 1 or condition_relation == 0:
            wr = " AND "
        elif condition_relation == 2:
            wr = " OR "

        table_info = self.conn.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].sql
        # table_info = table_info1
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(','):
            c, t = tup.split(' ')
            schema[c] = t

        tmp = ""
        for sel, agg in zip(select_index, aggregation_index):
            select_str = 'col_{}'.format(sel + 1)
            agg_str = agg_dict[agg]
            if agg:
                tmp += '{}({}),'.format(agg_str, select_str)
            else:
                tmp += '({}),'.format(select_str)
        tmp = tmp[:-1]

        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and (isinstance(val, str) or isinstance(val, str)):
                val = val.lower()
            if schema['col_{}'.format(col_index + 1)] == 'real' and not isinstance(val, (int, float)):
                try:
                    val = float(parse_decimal(val, locale='en_US'))
                except NumberFormatError as e:
                    try:
                        val = float(num_re.findall(val)[0])  # need to understand an d debug this part.
                    except:
                        # Although column is of number, selected one is not number. Do nothing in this case.
                        pass
            where_clause.append('col_{} {} :col_{}'.format(col_index + 1, cond_op_dict[op], col_index + 1))
            where_map['col_{}'.format(col_index + 1)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + wr.join(where_clause)
        query = 'SELECT {} FROM {} {}'.format(tmp, table_id, where_str)
        out = self.conn.query(query, **where_map)
        return out.as_dict()

    def execute_return_query(self, table_id, select_index, aggregation_index, conditions, lower=True):
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        table_info = self.db.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].sql.replace('\n','')
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t
        select = 'col{}'.format(select_index)
        agg = agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and (isinstance(val, str) or isinstance(val, str)):
                val = val.lower()
            if schema['col{}'.format(col_index)] == 'real' and not isinstance(val, (int, float)):
                try:
                    # print('!!!!!!value of val is: ', val, 'type is: ', type(val))
                    # val = float(parse_decimal(val)) # somehow it generates error.
                    val = float(parse_decimal(val, locale='en_US'))
                    # print('!!!!!!After: val', val)

                except NumberFormatError as e:
                    val = float(num_re.findall(val)[0])
            where_clause.append('col{} {} :col{}'.format(col_index, cond_ops[op], col_index))
            where_map['col{}'.format(col_index)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        query = 'SELECT {} AS result FROM {} {}'.format(select, table_id, where_str)
        #print query
        out = self.db.query(query, **where_map)


        return [o.result for o in out], query
    def show_table(self, table_id):
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        rows = self.db.query('select * from ' +table_id)
        print(rows.dataset)