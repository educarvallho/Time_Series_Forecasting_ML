import re


def export_model_to_ONNX(**kwargs):
    model = kwargs.get("model")
    symbol = kwargs.get("symbol")
    periods = kwargs.get("periods")
    periods_meta = kwargs.get("periods_meta")
    model_number = kwargs.get("model_number")
    export_path = kwargs.get("export_path")

    model[1].save_model(
        export_path + "catmodel " + symbol + " " + str(model_number) + ".onnx",
        format="onnx",
        export_parameters={
            "onnx_domain": "ai.catboost",
            "onnx_model_version": 1,
            "onnx_doc_string": "main model",
            "onnx_graph_name": "CatBoostModel_main",
        },
        pool=None,
    )

    model[2].save_model(
        export_path + "catmodel_m " + symbol + " " + str(model_number) + ".onnx",
        format="onnx",
        export_parameters={
            "onnx_domain": "ai.catboost",
            "onnx_model_version": 1,
            "onnx_doc_string": "meta model",
            "onnx_graph_name": "CatBoostModel_meta",
        },
        pool=None,
    )

    code = "#include <Math\Stat\Math.mqh>"
    code += "\n"
    code += (
        '#resource "catmodel '
        + symbol
        + " "
        + str(model_number)
        + '.onnx" as uchar ExtModel_'
        + symbol
        + "_"
        + str(model_number)
        + "[]"
    )
    code += "\n"
    code += (
        '#resource "catmodel_m '
        + symbol
        + " "
        + str(model_number)
        + '.onnx" as uchar ExtModel2_'
        + symbol
        + "_"
        + str(model_number)
        + "[]"
    )
    code += "\n\n"
    code += (
        "int Periods"
        + symbol
        + "_"
        + str(model_number)
        + "["
        + str(len(periods))
        + "] = {"
        + ",".join(map(str, periods))
        + "};"
    )
    code += "\n"
    code += (
        "int Periods_m"
        + symbol
        + "_"
        + str(model_number)
        + "["
        + str(len(periods_meta))
        + "] = {"
        + ",".join(map(str, periods_meta))
        + "};"
    )
    code += "\n\n"

    # get features
    code += (
        "void fill_arays"
        + symbol
        + "_"
        + str(model_number)
        + "( double &features[]) {\n"
    )
    code += "   double pr[], ret[];\n"
    code += "   ArrayResize(ret, 1);\n"
    code += (
        "   for(int i=ArraySize(Periods"
        + symbol
        + "_"
        + str(model_number)
        + ")-1; i>=0; i--) {\n"
    )
    code += (
        "       CopyClose(NULL,PERIOD_H1,1,Periods"
        + symbol
        + "_"
        + str(model_number)
        + "[i],pr);\n"
    )
    code += "       ret[0] = MathMean(pr);\n"
    code += (
        "       ArrayInsert(features, ret, ArraySize(features), 0, WHOLE_ARRAY); }\n"
    )
    code += "   ArraySetAsSeries(features, true);\n"
    code += "}\n\n"

    # get features
    code += (
        "void fill_arays_m"
        + symbol
        + "_"
        + str(model_number)
        + "( double &features[]) {\n"
    )
    code += "   double pr[], ret[];\n"
    code += "   ArrayResize(ret, 1);\n"
    code += (
        "   for(int i=ArraySize(Periods_m"
        + symbol
        + "_"
        + str(model_number)
        + ")-1; i>=0; i--) {\n"
    )
    code += (
        "       CopyClose(NULL,PERIOD_H1,1,Periods_m"
        + symbol
        + "_"
        + str(model_number)
        + "[i],pr);\n"
    )
    code += "       ret[0] = MathSkewness(pr);\n"
    code += (
        "       ArrayInsert(features, ret, ArraySize(features), 0, WHOLE_ARRAY); }\n"
    )
    code += "   ArraySetAsSeries(features, true);\n"
    code += "}\n\n"

    file = open(
        export_path + str(symbol) + " ONNX include" + " " + str(model_number) + ".mqh",
        "w",
    )
    file.write(code)

    file.close()
    print("The file " + "ONNX include" + ".mqh " + "has been written to disk")


def remove_inner_braces_and_second_bracket(text):
    # Regular expression for searching for double LeafValues[N][1] = { ... };
    pattern = re.compile(r"(double LeafValues\[\d+\]\[1\] = \{)(.*?)(\};)", re.DOTALL)

    # Function for replacing internal curly brackets and removing the second square bracket
    def replace_inner_braces_and_second_bracket(match):
        inner_content = match.group(2)
        # Remove internal curly brackets
        inner_content = re.sub(r"\{([^{}]*)\}", r"\1", inner_content)
        # Remove the second square bracket
        return match.group(1).replace("[1]", "") + inner_content + match.group(3)

    # Replace internal curly brackets and remove the second square bracket
    result = pattern.sub(replace_inner_braces_and_second_bracket, text)

    return result


def export_model_to_MQL4_code(**kwargs):
    model = kwargs.get("model")
    symbol = kwargs.get("symbol")
    periods = kwargs.get("periods")
    periods_meta = kwargs.get("periods_meta")
    model_number = kwargs.get("model_number")
    export_path = kwargs.get("export_path")

    model[1].save_model("catmodel.h", format="cpp", export_parameters=None, pool=None)
    model[2].save_model(
        "meta_catmodel.h", format="cpp", export_parameters=None, pool=None
    )

    # add variables
    code = (
        "int Periods"
        + "["
        + str(len(periods))
        + "] = {"
        + ",".join(map(str, periods))
        + "};"
    )
    code += "\n"
    code += (
        "int Periods_m"
        + "["
        + str(len(periods_meta))
        + "] = {"
        + ",".join(map(str, periods_meta))
        + "};"
    )
    code += "\n\n"

    # get features
    code += "void fill_arays" + "( double &features[]) {\n"
    code += "   double pr[];\n"
    code += "   ArrayResize(features, ArraySize(Periods));\n"
    code += "   for(int i=ArraySize(Periods)-1; i>=0; i--) {\n"
    code += "       int copyed = CopyClose(NULL,PERIOD_H1,1,Periods[i],pr);\n"
    code += "       if (copyed != Periods[i]) break;\n"
    code += "       features[i] = MathMean(pr);\n"
    code += "}\n"
    code += "}\n\n"

    # get features
    code += "void fill_arays_m" + "( double &features[]) {\n"
    code += "   double pr[];\n"
    code += "   ArrayResize(features, ArraySize(Periods_m));\n"
    code += "   for(int i=ArraySize(Periods_m)-1; i>=0; i--) {\n"
    code += "       int copyed = CopyClose(NULL,PERIOD_H1,1,Periods_m[i],pr);\n"
    code += "       if (copyed != Periods_m[i]) break;\n"
    code += "       features[i] = MathSkewness(pr);\n"
    code += "}\n"
    code += "}\n\n"

    # add CatBosst base model
    code += (
        "double catboost_model" + str(model_number) + "(const double &features[]) { \n"
    )
    code += "    "
    with open("catmodel.h", "r") as file:
        data = file.read()
        parsed_model_tree = data[
            data.find("unsigned int TreeDepth") : data.find("double Scale = 1;")
        ]
        code += remove_inner_braces_and_second_bracket(parsed_model_tree)
    code += "\n\n"
    code += (
        "return "
        + "ApplyCatboostModel"
        + str(model_number)
        + "(features, TreeDepth, TreeSplits , BorderCounts, Borders, LeafValues); } \n\n"
    )

    # add CatBosst meta model
    code += (
        "double catboost_meta_model"
        + str(model_number)
        + "(const double &features[]) { \n"
    )
    code += "    "
    with open("meta_catmodel.h", "r") as file:
        data = file.read()
        parsed_model_tree = data[
            data.find("unsigned int TreeDepth") : data.find("double Scale = 1;")
        ]
        code += remove_inner_braces_and_second_bracket(parsed_model_tree)
    code += "\n\n"
    code += (
        "return "
        + "ApplyCatboostModel"
        + str(model_number)
        + "(features, TreeDepth, TreeSplits , BorderCounts, Borders, LeafValues); } \n\n"
    )

    code += (
        "double ApplyCatboostModel"
        + str(model_number)
        + "(const double &features[],uint &TreeDepth_[],uint &TreeSplits_[],uint &BorderCounts_[],float &Borders_[],double &LeafValues_[]) {\n\
    uint FloatFeatureCount=ArrayRange(BorderCounts_,0);\n\
    uint BinaryFeatureCount=ArrayRange(Borders_,0);\n\
    uint TreeCount=ArrayRange(TreeDepth_,0);\n\
    bool     binaryFeatures[];\n\
    ArrayResize(binaryFeatures,BinaryFeatureCount);\n\
    uint binFeatureIndex=0;\n\
    for(uint i=0; i<FloatFeatureCount; i++) {\n\
       for(uint j=0; j<BorderCounts_[i]; j++) {\n\
          binaryFeatures[binFeatureIndex]=features[i]>Borders_[binFeatureIndex];\n\
          binFeatureIndex++;\n\
       }\n\
    }\n\
    double result=0.0;\n\
    uint treeSplitsPtr=0;\n\
    uint leafValuesForCurrentTreePtr=0;\n\
    for(uint treeId=0; treeId<TreeCount; treeId++) {\n\
       uint currentTreeDepth=TreeDepth_[treeId];\n\
       uint index=0;\n\
       for(uint depth=0; depth<currentTreeDepth; depth++) {\n\
          index|=(binaryFeatures[TreeSplits_[treeSplitsPtr+depth]]<<depth);\n\
       }\n\
       result+=LeafValues_[leafValuesForCurrentTreePtr+index];\n\
       treeSplitsPtr+=currentTreeDepth;\n\
       leafValuesForCurrentTreePtr+=(1<<currentTreeDepth);\n\
    }\n\
    return 1.0/(1.0+MathPow(M_E,-result));\n\
    }\n\n"
    )

    file = open(
        export_path + str(symbol) + "_model_MQL_code_" + str(model_number) + ".mqh", "w"
    )
    file.write(code)

    file.close()
    print("The file " + "cat_model" + ".mqh " + "has been written to disc")
