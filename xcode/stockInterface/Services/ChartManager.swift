//
//  ChartManager.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/12/23.
//

import Cocoa

class ChartManager {
    
    static let shared = ChartManager()
    
    weak var currentStockView: StockView?
    
    var currentAggregate: StockAggregate?
    var indicatorData: IndicatorData?
    
    func chartStock(_ aggregate: StockAggregate) {
        
        self.currentAggregate = aggregate
        currentStockView?.stockAggregate = aggregate
    }
    
    func setIndicatorData(_ indicatorData: IndicatorData) {
        self.indicatorData = indicatorData
        currentStockView?.addColoredLines(lines: [(indicatorData.ema14, CGColor.ChartEMALines.NSColor())])
        currentStockView?.addColoredLines(lines: [(indicatorData.ema28, CGColor.ChartEMALines.NSColor())])
        currentStockView?.addColoredLines(lines: [(indicatorData.sma50, CGColor.LightSMALine.NSColor())])
        currentStockView?.addColoredLines(lines: [(indicatorData.sma200, CGColor.DarkSMALine.NSColor())])
        currentStockView?.mouseDelegate = self
    }
    
    func setAuxSets(auxSets: [StockViewAuxGraphProperties]) {
        currentStockView?.setAuxViews(auxSets)
    }
}

extension ChartManager: StockViewMouseDelegate {
    func candleHoverChanged(index: Int) {
        guard let data = indicatorData else {return}
        guard let aggregate = currentAggregate else {return}
        if index < 0 || index >= data.length {return}
        
        LabelValueController.shared.setLabelValue(index: 0, label: "Open", value: aggregate.candles[index].open.toRoundedString(precision: 2))
        LabelValueController.shared.setLabelValue(index: 1, label: "Close", value: aggregate.candles[index].close.toRoundedString(precision: 2))
        LabelValueController.shared.setLabelValue(index: 2, label: "High", value: aggregate.candles[index].high.toRoundedString(precision: 2))
        LabelValueController.shared.setLabelValue(index: 3, label: "Low", value: aggregate.candles[index].low.toRoundedString(precision: 2))
        
        
        LabelValueController.shared.setLabelValue(index: 4, label: "SMA 200", value: data.sma200[index].toRoundedString(precision: 2))
        LabelValueController.shared.setLabelValue(index: 5, label: "SMA 50", value: data.sma50[index].toRoundedString(precision: 2))
        LabelValueController.shared.setLabelValue(index: 6, label: "EMA 28", value: data.ema28[index].toRoundedString(precision: 2))
        LabelValueController.shared.setLabelValue(index: 7, label: "EMA 14", value: data.ema14[index].toRoundedString(precision: 2))
    }
}
